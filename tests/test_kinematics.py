import math

import numpy as np

from mechverify.kinematics import FourBarGeometry


def test_loop_closure_residual():
    fb = FourBarGeometry(1.0, 2.5, 2.5, 2.5, coupler_p=1.25)
    sol = fb.solve_initial(0.0, branch=1)
    assert sol is not None
    t3, t4 = sol
    assert fb.loop_residual(0.0, t3, t4) < 1e-9


def test_two_assembly_branches():
    fb = FourBarGeometry(1.0, 2.5, 2.5, 2.5)
    assert len(fb.theta4_candidates(0.0)) == 2


def test_sweep_completes_and_transmission_range():
    fb = FourBarGeometry(1.0, 2.5, 2.5, 2.5, coupler_p=1.25)
    sw = fb.sweep(0.0, 360.0, 720, branch=1)
    assert sw.completed
    mu = sw.mu_deg[sw.valid]
    assert np.all(mu >= -1e-9)
    assert np.all(mu <= 90.0 + 1e-6)


def test_continuation_stays_on_branch():
    fb = FourBarGeometry(1.0, 2.5, 2.5, 2.5)
    t3, t4 = fb.solve_initial(0.1, branch=1)
    res = fb.solve_continuation(0.11, t4)
    assert res is not None
    _, t4b = res
    assert abs(t4b - t4) < 0.2
