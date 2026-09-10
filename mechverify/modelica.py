"""Layer 2 (precise): Modelica code generation + consistent initial conditions.

The generator emits a *self-contained* Modelica model (no external library
dependency) that encodes the same planar constraint formulation used by
:mod:`mechverify.multibody`:

* each movable link has states ``<link>_x, <link>_y, <link>_th``
* revolute joints  -> anchor-coincidence equations
* prismatic joints -> orientation + perpendicular-offset equations
* the driver       -> one relative-angle equation driven by ``time``

Consistent initial conditions are taken from the Layer-1 solver so the model
starts on the same assembly branch.

The simulation backend (:func:`simulate`) calls OpenModelica's ``omc`` if it is
installed; otherwise it raises a clear error.  The code generation itself is
independent of the toolchain and fully testable.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from typing import Dict, List, Optional

import numpy as np

from .ir import Mechanism, Joint
from .kinematics import four_bar_from_mechanism

OMEGA = 2.0 * math.pi  # default driver speed: 1 revolution per unit time


# --------------------------------------------------------------- helpers
def _ident(s: str) -> str:
    return re.sub(r"[^0-9a-zA-Z_]", "_", s)


def ensure_general(mech: Mechanism) -> Mechanism:
    """Return a mechanism with explicit joint ``anchors`` (convert four-bar)."""
    if any(j.anchors for j in mech.joints):
        return mech

    fb = four_bar_from_mechanism(mech)
    ground = mech.ground
    driver_link = (mech.driver or {}).get("link")
    adj = []
    for j in mech.ground_joints():
        adj.append([x for x in j.links if x != ground][0])
    if driver_link not in adj:
        driver_link = adj[0]
    rocker = [x for x in adj if x != driver_link][0]
    coupler = [x for x in mech.link_ids() if x not in (ground, driver_link, rocker)][0]

    gj = mech.ground_joints()
    pos = {}
    for j in gj:
        other = [x for x in j.links if x != ground][0]
        pos[other] = list(j.position)
    A = pos[driver_link]
    D = pos[rocker]

    def mk(jid, typ, la, lb, aa, ab, axis=None):
        j = Joint(id=jid, type=typ, links=[la, lb], anchors={la: list(aa), lb: list(ab)})
        if axis:
            j.axis = {la: list(axis[0]), lb: list(axis[1])}
        return j

    joints = [
        mk("A", "revolute", ground, driver_link, A, [0.0, 0.0]),
        mk("B", "revolute", driver_link, coupler, [fb.a, 0.0], [0.0, 0.0]),
        mk("C", "revolute", coupler, rocker, [fb.b, 0.0], [0.0, 0.0]),
        mk("D", "revolute", rocker, ground, [fb.c, 0.0], D),
    ]
    poi = None
    if mech.coupler_point is not None:
        p, phi = fb.coupler_p, fb.coupler_phi
        poi = {"link": coupler, "local": [p * math.cos(phi), p * math.sin(phi)]}
    elif mech.point_of_interest is not None:
        poi = mech.point_of_interest

    return Mechanism(
        id=mech.id,
        links=mech.links,
        joints=joints,
        ground=ground,
        dimensions=mech.dimensions,
        point_of_interest=poi,
        driver=mech.driver,
    )


def initial_state(mech: Mechanism, driver_start_deg: float, seed: int = 0) -> Dict[str, List[float]]:
    """Solve the reference configuration and return {link: [x, y, theta]}."""
    gen = ensure_general(mech)
    from .multibody import GeneralPlanarMechanism

    solver = GeneralPlanarMechanism(gen)
    q = solver.solve_multistart(math.radians(driver_start_deg), seed=seed)
    if q is None:
        raise RuntimeError("could not find a consistent initial configuration")
    state = {}
    for link in solver.movable:
        i = solver.index[link] * 3
        state[link] = [float(q[i]), float(q[i + 1]), float(q[i + 2])]
    return state


# --------------------------------------------------------------- generation
def _world_expr(link: str, ground: str, ax: float, ay: float):
    if link == ground:
        return ("(%.12g)" % ax, "(%.12g)" % ay)
    v = _ident(link)
    return (
        "%s_x + (%.12g)*cos(%s_th) - (%.12g)*sin(%s_th)" % (v, ax, v, ay, v),
        "%s_y + (%.12g)*sin(%s_th) + (%.12g)*cos(%s_th)" % (v, ax, v, ay, v),
    )


def generate_modelica(
    mech: Mechanism,
    driver_start_deg: Optional[float] = None,
    driver_omega: float = OMEGA,
    stop_time: float = 1.0,
    interval: float = 0.002,
) -> str:
    gen = ensure_general(mech)
    ground = gen.ground
    movable = [l.id for l in gen.links if l.id != ground]

    sw = dict(gen.driver or {})
    if driver_start_deg is None:
        driver_start_deg = float(sw.get("start", 0.0))
    state = initial_state(gen, driver_start_deg, seed=int(sw.get("seed", 0)))

    dj = next(j for j in gen.joints if j.id == gen.driver["joint"])
    la, lb = dj.links

    model = _ident(gen.id)
    ir_json = json.dumps(gen.to_dict(), ensure_ascii=False, separators=(",", ":"))
    L: List[str] = []
    L.append("model %s" % model)
    L.append('  // generated by mechverify -- self-contained planar mechanism model')
    L.append("  // mechverify-ir: %s" % ir_json)
    L.append("  parameter Real driver_omega = %.12g;" % driver_omega)
    L.append("  Real driver_angle;")
    for link in movable:
        v = _ident(link)
        x, y, th = state[link]
        L.append("  Real %s_x(start=%.12g);" % (v, x))
        L.append("  Real %s_y(start=%.12g);" % (v, y))
        L.append("  Real %s_th(start=%.12g);" % (v, th))
    if gen.point_of_interest:
        L.append("  output Real poi_x;")
        L.append("  output Real poi_y;")
    L.append("equation")

    # driver
    start_rad = math.radians(driver_start_deg)
    L.append("  driver_angle = %.12g + driver_omega*time;" % start_rad)
    th_a = "0.0" if la == ground else "%s_th" % _ident(la)
    th_b = "0.0" if lb == ground else "%s_th" % _ident(lb)
    L.append("  (%s) - (%s) = driver_angle;" % (th_b, th_a))

    # joints
    for j in gen.joints:
        ja, jb = j.links
        if j.type == "revolute":
            ax, ay = _world_expr(ja, ground, *j.anchors[ja])
            bx, by = _world_expr(jb, ground, *j.anchors[jb])
            L.append("  0.0 = (%s) - (%s);" % (ax, bx))
            L.append("  0.0 = (%s) - (%s);" % (ay, by))
        elif j.type == "prismatic":
            aa = j.axis[ja]
            ab = j.axis[jb]
            ang_a = math.atan2(aa[1], aa[0])
            ang_b = math.atan2(ab[1], ab[0])
            ta = "0.0" if ja == ground else "%s_th" % _ident(ja)
            tb = "0.0" if jb == ground else "%s_th" % _ident(jb)
            L.append("  ((%s) + %.12g) - ((%s) + %.12g) = 0.0;" % (tb, ang_b, ta, ang_a))
            # axis direction in world (link a)
            cth = "1.0" if ja == ground else "cos(%s_th)" % _ident(ja)
            sth = "0.0" if ja == ground else "sin(%s_th)" % _ident(ja)
            axis_x = "((%.12g)*(%s) - (%.12g)*(%s))" % (aa[0], cth, aa[1], sth)
            axis_y = "((%.12g)*(%s) + (%.12g)*(%s))" % (aa[0], sth, aa[1], cth)
            pax, pay = _world_expr(ja, ground, *j.anchors[ja])
            pbx, pby = _world_expr(jb, ground, *j.anchors[jb])
            L.append("  0.0 = (%s)*((%s) - (%s)) - (%s)*((%s) - (%s));"
                     % (axis_x, pby, pay, axis_y, pbx, pax))

    # point of interest
    if gen.point_of_interest:
        link = gen.point_of_interest["link"]
        lx, ly = gen.point_of_interest["local"]
        px, py = _world_expr(link, ground, lx, ly)
        L.append("  poi_x = %s;" % px)
        L.append("  poi_y = %s;" % py)

    L.append("  annotation(experiment(StopTime=%.12g, Interval=%.12g));" % (stop_time, interval))
    L.append("end %s;" % model)
    return "\n".join(L) + "\n"


def write_model(mech: Mechanism, path: str, **kwargs) -> str:
    code = generate_modelica(mech, **kwargs)
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)
    return code


# --------------------------------------------------------------- backend
def _find_omc() -> Optional[str]:
    exe = shutil.which("omc")
    if exe:
        return exe
    import glob

    patterns = [
        r"C:\Program Files\OpenModelica*\bin\omc.exe",
        r"C:\OpenModelica*\bin\omc.exe",
        r"D:\OpenModelica*\bin\omc.exe",
        r"D:\Program Files\OpenModelica*\bin\omc.exe",
    ]
    for pat in patterns:
        hits = glob.glob(pat)
        if hits:
            return hits[0]
    return None


def omc_available() -> bool:
    return _find_omc() is not None


def simulate(
    mo_path: str,
    model_name: str,
    stop_time: float = 1.0,
    number_of_intervals: int = 500,
    tolerance: Optional[float] = None,
    timeout: int = 300,
) -> str:
    """Run an OpenModelica simulation, returning the result CSV path.

    Raises ``RuntimeError`` if ``omc`` is not on PATH.
    """
    omc = _find_omc()
    if omc is None:
        raise RuntimeError(
            "OpenModelica 'omc' not found. Install OpenModelica to run the precise "
            "layer; code generation and initial conditions work without it."
        )
    workdir = os.path.dirname(os.path.abspath(mo_path))
    mos = tempfile.NamedTemporaryFile("w", suffix=".mos", delete=False, encoding="utf-8")
    mos.write('loadFile("%s"); getErrorString();\n' % os.path.abspath(mo_path).replace("\\", "/"))
    sim_cmd = 'simulate(%s, stopTime=%.12g, numberOfIntervals=%d, outputFormat="csv"' % (
        model_name,
        stop_time,
        number_of_intervals,
    )
    if tolerance is not None:
        sim_cmd += ", tolerance=%.3g" % tolerance
    sim_cmd += "); getErrorString();\n"
    mos.write(sim_cmd)
    mos.close()
    env = dict(os.environ)
    bin_dir = os.path.dirname(omc)
    env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")
    env["OPENMODELICAHOME"] = os.path.dirname(bin_dir)
    proc = subprocess.run(
        [omc, mos.name],
        cwd=workdir,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        env=env,
    )
    os.unlink(mos.name)
    if proc.returncode != 0:
        raise RuntimeError("omc failed:\n%s\n%s" % (proc.stdout, proc.stderr))
    csv = os.path.join(workdir, "%s_res.csv" % model_name)
    if not os.path.exists(csv):
        raise RuntimeError("simulation produced no result file; omc output:\n%s" % proc.stdout)
    return csv
