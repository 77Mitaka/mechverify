"""Generate the figures for the paper from the actual mechverify results."""

import os
import sys
import csv
import math

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = None
for _cand in (os.path.dirname(HERE), os.path.join(os.path.dirname(HERE), "mechverify")):
    if os.path.isdir(os.path.join(_cand, "mechverify")):
        PROJ = _cand
        break
if PROJ is None:
    raise RuntimeError("cannot locate the mechverify package")
sys.path.insert(0, PROJ)

from mechverify import Mechanism
from mechverify.kinematics import four_bar_from_mechanism
from mechverify.multibody import GeneralPlanarMechanism
from mechverify.modelica import ensure_general, simulate
from mechverify.robustness import perturb_mechanism
from mechverify.requirement import path_straightness

EX = os.path.join(PROJ, "examples")


def _convex(pts):
    signs = []
    n = len(pts)
    for i in range(n):
        p0, p1, p2 = pts[i], pts[(i + 1) % n], pts[(i + 2) % n]
        v1, v2 = p1 - p0, p2 - p1
        signs.append(np.sign(v1[0] * v2[1] - v1[1] * v2[0]))
    return abs(sum(signs)) == n


def draw_fourbar(ax, fb, sw, k):
    t2, t3, t4 = float(sw.theta2[k]), float(sw.theta3[k]), float(sw.theta4[k])
    A = np.array([0.0, 0.0])
    D = np.array([fb.d, 0.0])
    B = A + fb.a * np.array([math.cos(t2), math.sin(t2)])
    C = D + fb.c * np.array([math.cos(t4), math.sin(t4)])
    P = np.array(fb.coupler_point(t2, t3))
    ax.plot([A[0], D[0]], [A[1], D[1]], "k--", lw=1.0)
    for p, q in [(A, B), (B, C), (C, D), (B, P), (C, P)]:
        ax.plot([p[0], q[0]], [p[1], q[1]], "-", color="0.25", lw=2.0)
    for p in (A, B, C, D):
        ax.plot(p[0], p[1], "ko", ms=3.5)
    ax.plot(P[0], P[1], "o", color="red", ms=6, zorder=5)
    ax.annotate("P", P, textcoords="offset points", xytext=(6, 4), color="red", fontsize=11)


def draw_config(ax, gen, gpm, q):
    jpos = {}
    for j in gen.joints:
        jpos[j.id] = gpm.world_point(q, j.links[0], j.anchors[j.links[0]])
    for l in gen.links:
        pts = [jpos[j.id] for j in gen.joints if l.id in j.links]
        if len(pts) >= 2:
            pts = np.array(pts)
            style = "k--" if l.id == gen.ground else "-"
            lw = 1.0 if l.id == gen.ground else 2.0
            col = "k" if l.id == gen.ground else "0.25"
            ax.plot(pts[:, 0], pts[:, 1], style, lw=lw, color=col)
    poi = gpm.poi(q)
    poi_link = gen.point_of_interest["link"]
    for j in gen.joints:
        if poi_link in j.links:
            lp = jpos[j.id]
            if (lp[0] - poi[0]) ** 2 + (lp[1] - poi[1]) ** 2 > 1e-12:
                ax.plot([lp[0], poi[0]], [lp[1], poi[1]], "-", color="0.25", lw=1.2)
    for jid, p in jpos.items():
        ax.plot(p[0], p[1], "ko", ms=3.5)
    ax.plot(poi[0], poi[1], "o", color="red", ms=6, zorder=5)
    ax.annotate("P", poi, textcoords="offset points", xytext=(6, 4), color="red", fontsize=11)


def pick_open_config(gen, gpm, sw):
    best_k, best_area = None, -1.0
    for k in np.where(sw.valid)[0]:
        q = sw.configs[k]
        pts = np.array([gpm.world_point(q, j.links[0], j.anchors[j.links[0]]) for j in gen.joints])
        area = np.ptp(pts[:, 0]) * np.ptp(pts[:, 1])
        if area > best_area:
            best_area, best_k = area, int(k)
    return best_k


def fig_mechanisms():
    cases = [
        ("fourbar_crank_rocker.json", "四杆曲柄摇杆"),
        ("fourbar_straightline.json", "直线四杆机构"),
        ("slider_crank.json", "曲柄滑块"),
        ("sixbar_watt.json", "六杆(Watt)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 7.2))
    for ax, (name, title) in zip(axes.ravel(), cases):
        mech = Mechanism.from_json_file(os.path.join(EX, name))
        if not any(j.anchors for j in mech.joints):
            fb = four_bar_from_mechanism(mech)
            sw = fb.sweep(0.0, 360.0, 240, branch=1)
            A = np.array([0.0, 0.0])
            D = np.array([fb.d, 0.0])
            k = None
            for kk in np.where(sw.valid)[0]:
                t2, t4 = float(sw.theta2[kk]), float(sw.theta4[kk])
                B = A + fb.a * np.array([math.cos(t2), math.sin(t2)])
                C = D + fb.c * np.array([math.cos(t4), math.sin(t4)])
                if _convex([A, B, C, D]):
                    k = int(kk)
                    break
            if k is None:
                v = np.where(sw.valid)[0]
                k = int(v[len(v) // 2])
            draw_fourbar(ax, fb, sw, k)
            pts = sw.valid_points()
        else:
            gen = ensure_general(mech)
            gpm = GeneralPlanarMechanism(gen)
            d0 = float((mech.driver or {}).get("start", 0.0))
            sw = gpm.sweep(d0, 360.0, 240)
            k = pick_open_config(gen, gpm, sw)
            draw_config(ax, gen, gpm, sw.configs[k])
            pts = sw.points
        ax.plot(pts[:, 0], pts[:, 1], "-", color="tab:red", lw=1.2, label="连杆点 P 轨迹")
        ax.set_title(title, fontsize=11)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out = os.path.join(HERE, "fig_mechanisms.pdf")
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


def fig_crossval():
    mech = Mechanism.from_json_file(os.path.join(EX, "slider_crank.json"))
    from mechverify.planar_mechanics import generate_planar_mechanics

    import tempfile

    d = tempfile.mkdtemp(prefix="fig_crossval_")
    mo = os.path.join(d, "slider_crank.mo")
    with open(mo, "w", encoding="utf-8") as f:
        f.write(generate_planar_mechanics(mech, stop_time=1.0))
    path = simulate(mo, "slider_crank", stop_time=1.0, number_of_intervals=200)
    with open(path) as f:
        rows = list(csv.DictReader(f))
    t = np.array([float(r["time"]) for r in rows])
    x = np.array([float(r["poi_x"]) for r in rows])
    th = math.pi / 2 + 2 * math.pi * t
    xa = np.cos(th) + np.sqrt(2.5 ** 2 - np.sin(th) ** 2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.4))
    ax1.plot(t, xa, "-", color="k", lw=2, label="解析解")
    ax1.plot(t, x, "--", color="tab:red", lw=1.2, label="PlanarMechanics 仿真")
    ax1.set_xlabel("时间 t (s)")
    ax1.set_ylabel("滑块点 $P$ 位置 $x$")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    err = np.abs(x - xa)
    ax2.semilogy(t, np.maximum(err, 1e-16), color="tab:blue")
    ax2.set_xlabel("时间 t (s)")
    ax2.set_ylabel("绝对误差 $|x-\\hat{x}|$")
    ax2.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    out = os.path.join(HERE, "fig_crossval.pdf")
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


def fig_robustness():
    mech = Mechanism.from_json_file(os.path.join(EX, "fourbar_straightline.json"))
    rng = np.random.default_rng(0)
    vals = []
    for _ in range(150):
        m = perturb_mechanism(mech, 0.002, rng)
        sw = four_bar_from_mechanism(m).sweep(0.0, 360.0, 360, branch=1)
        vals.append(path_straightness(sw.valid_points(), 0.3))
    vals = np.array(vals)
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.hist(vals, bins=25, color="tab:blue", alpha=0.75)
    ax.axvline(0.005, color="k", ls="--", lw=1.5, label="需求阈值 0.005")
    ax.set_xlabel("直线度(相对偏差)")
    ax.set_ylabel("样本数")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(HERE, "fig_robustness.pdf")
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    fig_mechanisms()
    fig_crossval()
    fig_robustness()
