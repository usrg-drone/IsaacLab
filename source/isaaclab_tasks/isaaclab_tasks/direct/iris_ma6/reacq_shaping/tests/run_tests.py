#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone test suite for the recovery-shaping reward (ticket 050, Slice C)."""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run reacq_shaping test suite")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import sys
import traceback
import torch
from datetime import datetime

from isaaclab_tasks.direct.iris_ma6.reacq_shaping import ReacqShaper, ReacqShapingCfg

_RESULT_PATH = __file__.rsplit("/", 1)[0] + "/test_result.txt"
_RESULT = open(_RESULT_PATH, "w")


class TestResults:
    def __init__(self):
        self.passed = []
        self.failed = []
        self.errors = []

    def _emit(self, s):
        sys.stdout.write(s + "\n"); sys.stdout.flush()
        _RESULT.write(s + "\n"); _RESULT.flush()

    def add_pass(self, name):
        self.passed.append(name); self._emit(f"  PASS {name}")

    def add_fail(self, name, err):
        self.failed.append((name, err)); self._emit(f"  FAIL {name}")
        for line in str(err).split("\n")[:6]:
            self._emit(f"      {line}")

    def add_error(self, name, err):
        self.errors.append((name, err)); self._emit(f"  ERROR {name}")
        self._emit(f"      {err}")

    def header(self, s):
        self._emit("\n" + "=" * 80); self._emit(s); self._emit("=" * 80)

    def print_summary(self):
        total = len(self.passed) + len(self.failed) + len(self.errors)
        self._emit("\n" + "=" * 80); self._emit("TEST SUMMARY"); self._emit("=" * 80)
        self._emit(f"Total: {total}  Passed: {len(self.passed)}  Failed: {len(self.failed)}  Errors: {len(self.errors)}")
        ok = not self.failed and not self.errors
        self._emit(f"\nRESULT: {'PASS' if ok else 'FAIL'}")
        return ok


def _inputs(boresight, target=(1.0, 0.0, 0.0), cam=(0.0, 0.0, 0.0), device="cpu"):
    """Build [N=1, A=1, 3] inputs for a single boresight vector and fixed target/cam."""
    b = torch.tensor([[boresight]], dtype=torch.float, device=device)       # [1,1,3]
    cam_pos = torch.tensor([[cam]], dtype=torch.float, device=device)       # [1,1,3]
    tgt = torch.tensor([target], dtype=torch.float, device=device)          # [1,3]
    return b, cam_pos, tgt


def run_tests(results: TestResults, device):
    results.header(f"reacq_shaping tests  (device={device})")
    AIM = [1.0, 0.0, 0.0]          # boresight at target -> Phi=1
    PERP = [0.0, 1.0, 0.0]         # perpendicular        -> Phi=0.5
    AWAY = [-1.0, 0.0, 0.0]        # opposite             -> Phi=0
    deficit_T = torch.ones(1, 1, dtype=torch.bool, device=device)
    deficit_F = torch.zeros(1, 1, dtype=torch.bool, device=device)

    # 1) Potential basics: Phi in [0,1], =1 aimed, 0.5 perp, 0 away.
    try:
        sh = ReacqShaper(ReacqShapingCfg(shaping_scale=1.0), 1, 1, device)
        b, c, t = _inputs(AIM, device=device); phi_aim = sh._potential(b, c, t).item()
        b, c, t = _inputs(PERP, device=device); phi_perp = sh._potential(b, c, t).item()
        b, c, t = _inputs(AWAY, device=device); phi_away = sh._potential(b, c, t).item()
        assert abs(phi_aim - 1.0) < 1e-5, f"aim Phi={phi_aim}"
        assert abs(phi_perp - 0.5) < 1e-5, f"perp Phi={phi_perp}"
        assert abs(phi_away - 0.0) < 1e-5, f"away Phi={phi_away}"
        results.add_pass(f"potential: aim=1 ({phi_aim:.3f}), perp=0.5 ({phi_perp:.3f}), away=0 ({phi_away:.3f})")
    except Exception as e:
        results.add_fail("potential basics", traceback.format_exc())

    # 2) Entry step gives F=0 (no spurious spike on the first in-region step).
    try:
        sh = ReacqShaper(ReacqShapingCfg(shaping_scale=1.0), 1, 1, device)
        b, c, t = _inputs(AIM, device=device)
        F0 = sh.compute_shaping(b, c, t, deficit_T, t=0.0)
        assert torch.allclose(F0, torch.zeros_like(F0)), f"entry F={F0.item()}"
        results.add_pass("entry step F=0 (no spike)")
    except Exception as e:
        results.add_fail("entry step F=0", traceback.format_exc())

    # 3) F = scale*(Phi' - Phi) (gamma=1) on consecutive in-deficit steps; >0 on re-aim.
    try:
        cfg = ReacqShapingCfg(shaping_scale=1.0, gamma=1.0)
        sh = ReacqShaper(cfg, 1, 1, device)
        b, c, t = _inputs(PERP, device=device)
        sh.compute_shaping(b, c, t, deficit_T, t=0.0)            # entry: Phi=0.5, F=0 (region_prev False)
        b, c, t = _inputs(AIM, device=device)
        F1 = sh.compute_shaping(b, c, t, deficit_T, t=1.0).item()  # Phi'=1.0, was-in-deficit -> active
        expected = 1.0 * (1.0 * 1.0 - 0.5)  # 0.5
        assert abs(F1 - expected) < 1e-5, f"F={F1}, expected {expected}"
        assert F1 > 0.0, f"re-aim should reward, F={F1}"
        results.add_pass(f"F = scale*(Phi'-Phi) on re-aim ({F1:.4f} > 0)")
    except Exception as e:
        results.add_fail("PBRS formula / re-aim positive", traceback.format_exc())

    # 4) Gate zeros F outside the deficit regime (even if Phi changes a lot).
    try:
        sh = ReacqShaper(ReacqShapingCfg(shaping_scale=1.0), 1, 1, device)
        b, c, t = _inputs(PERP, device=device)
        sh.compute_shaping(b, c, t, deficit_F, t=0.0)
        b, c, t = _inputs(AIM, device=device)
        F1 = sh.compute_shaping(b, c, t, deficit_F, t=1.0)
        assert torch.allclose(F1, torch.zeros_like(F1)), f"non-deficit F={F1.item()}"
        results.add_pass("gate: F=0 outside DEFICIT")
    except Exception as e:
        results.add_fail("gate zeros F", traceback.format_exc())

    # 5) Idempotency: a second call at the same t returns cached F and does NOT double-advance Phi_prev.
    try:
        sh = ReacqShaper(ReacqShapingCfg(shaping_scale=1.0, gamma=1.0), 1, 1, device)
        b, c, t = _inputs(PERP, device=device)
        sh.compute_shaping(b, c, t, deficit_T, t=0.0)            # entry, Phi_prev=0.5
        b, c, t = _inputs(AIM, device=device)
        Fa = sh.compute_shaping(b, c, t, deficit_T, t=1.0).item()  # advances Phi_prev->1.0 (F=0.5)
        Fb = sh.compute_shaping(b, c, t, deficit_T, t=1.0).item()  # same t -> cached, no advance
        assert abs(Fa - Fb) < 1e-7, f"cached mismatch {Fa} vs {Fb}"
        b, c, t = _inputs(AIM, device=device)
        Fc = sh.compute_shaping(b, c, t, deficit_T, t=2.0).item()  # Phi'=1, Phi_prev should be 1.0
        expected = 1.0 * 1.0 - 1.0  # 0.0 (no double-advance AND no holding tax at gamma=1)
        assert abs(Fc - expected) < 1e-5, f"Phi_prev double-advanced? Fc={Fc}, expected {expected}"
        results.add_pass(f"idempotent within a step (no double-advance; Fc={Fc:.4f})")
    except Exception as e:
        results.add_fail("idempotency", traceback.format_exc())

    # 6) reset_idx clears state: the next step after reset is an entry step (F=0).
    try:
        sh = ReacqShaper(ReacqShapingCfg(shaping_scale=1.0), 1, 1, device)
        b, c, t = _inputs(PERP, device=device)
        sh.compute_shaping(b, c, t, deficit_T, t=0.0)
        b, c, t = _inputs(AIM, device=device)
        sh.compute_shaping(b, c, t, deficit_T, t=1.0)            # would be nonzero
        sh.reset_idx(torch.tensor([0], device=device))
        b, c, t = _inputs(AIM, device=device)
        Fr = sh.compute_shaping(b, c, t, deficit_T, t=2.0)       # entry again after reset
        assert torch.allclose(Fr, torch.zeros_like(Fr)), f"post-reset F={Fr.item()}"
        results.add_pass("reset_idx: post-reset step F=0")
    except Exception as e:
        results.add_fail("reset clears state", traceback.format_exc())

    # 7) Ungated (pure PBRS): F nonzero every step after the first, regardless of deficit_mask.
    try:
        sh = ReacqShaper(ReacqShapingCfg(shaping_scale=1.0, gamma=1.0, gate_to_deficit=False), 1, 1, device)
        b, c, t = _inputs(PERP, device=device)
        sh.compute_shaping(b, c, t, deficit_F, t=0.0)           # entry, F=0
        b, c, t = _inputs(AIM, device=device)
        F1 = sh.compute_shaping(b, c, t, deficit_F, t=1.0).item()  # deficit False, but ungated -> active
        expected = 1.0 * 1.0 - 0.5  # 0.5
        assert abs(F1 - expected) < 1e-5, f"ungated F={F1}, expected {expected}"
        results.add_pass(f"ungated PBRS: active everywhere ({F1:.4f})")
    except Exception as e:
        results.add_fail("ungated pure PBRS", traceback.format_exc())

    # 8) 2-step realistic: a drone slews its boresight back onto a target over several deficit steps;
    #    cumulative F > 0 (recovery credited), and NO negative drip on the held-aim steps.
    try:
        cfg = ReacqShapingCfg(shaping_scale=2.0, gamma=1.0)
        sh = ReacqShaper(cfg, 1, 1, device)
        cam = (5.0, -3.0, 2.0)
        target = (12.0, 4.0, 0.0)
        # boresights sweeping from ~away toward the true bearing over 6 steps
        tv = torch.tensor(target, device=device) - torch.tensor(cam, device=device)
        tv = (tv / tv.norm()).tolist()
        away = [-tv[0], -tv[1], -tv[2]]
        F_hist, phi_hist = [], []
        for k in range(6):
            a = k / 5.0  # 0 -> 1 interpolation toward the true bearing
            bvec = [(1 - a) * away[j] + a * tv[j] for j in range(3)]
            b, c, t = _inputs(bvec, target=target, cam=cam, device=device)
            phi_hist.append(sh._potential(b, c, t).item())
            F = sh.compute_shaping(b, c, t, deficit_T, t=float(k))
            F_hist.append(F.item())
        results._emit(f"      [diag] phi={[round(x,3) for x in phi_hist]} F={[round(x,3) for x in F_hist]}")
        assert abs(F_hist[0]) < 1e-7, f"entry not zero: {F_hist[0]}"
        assert sum(F_hist) > 0.0, f"net re-aim reward should be positive: {sum(F_hist):.4f}"
        assert phi_hist[-1] > 0.99, f"final aimed Phi={phi_hist[-1]:.3f}"
        # The recovery (Phi 0->1) step must be CREDITED, and the held-aim steps after it must NOT be
        # taxed (gamma=1): no negative F anywhere in this monotone-improving slew.
        assert max(F_hist) > 0.0, f"recovery step not credited: {F_hist}"
        assert min(F_hist) > -1e-6, f"holding-aim steps taxed (v1 bug): {F_hist}"
        results.add_pass(f"2-step slew-to-aim: sum(F)={sum(F_hist):.3f}>0, no holding tax (min={min(F_hist):.3f})")
    except Exception as e:
        results.add_fail("2-step realistic", traceback.format_exc())

    # 9) Multi-env / multi-agent shapes and independence.
    try:
        N, A = 4, 3
        sh = ReacqShaper(ReacqShapingCfg(shaping_scale=1.0), N, A, device)
        b = torch.nn.functional.normalize(torch.randn(N, A, 3, device=device), dim=-1)
        c = torch.zeros(N, A, 3, device=device)
        tgt = torch.randn(N, 3, device=device)
        dm = torch.zeros(N, A, dtype=torch.bool, device=device); dm[0, 0] = True; dm[2, 1] = True
        sh.compute_shaping(b, c, tgt, dm, t=0.0)
        F = sh.compute_shaping(b, c, tgt, dm, t=1.0)
        assert F.shape == (N, A), f"shape {F.shape}"
        nonzero = (F.abs() > 0).nonzero(as_tuple=False)
        # Only the two deficit cells can be nonzero (and only because region_prev was True there).
        assert set(map(tuple, nonzero.tolist())) <= {(0, 0), (2, 1)}, f"nonzero at {nonzero.tolist()}"
        results.add_pass("multi-env/agent: shape + gate independence")
    except Exception as e:
        results.add_fail("multi-env/agent", traceback.format_exc())

    # 10) REGRESSION (v1 bug): holding aim during a deficit must NOT be taxed at gamma=1 (F=0).
    try:
        sh = ReacqShaper(ReacqShapingCfg(shaping_scale=10.0, gamma=1.0), 1, 1, device)
        b, c, t = _inputs(AIM, device=device)
        Fs = [sh.compute_shaping(b, c, t, deficit_T, t=float(k)).item() for k in range(5)]
        # Aim is constant (Phi=1) throughout a persistent deficit -> every step F=0 (no drip).
        assert all(abs(f) < 1e-6 for f in Fs), f"holding aim taxed at gamma=1 (v1 bug): {Fs}"
        results.add_pass(f"no holding tax at gamma=1 (constant-aim deficit F={[round(f,4) for f in Fs]})")
    except Exception as e:
        results.add_fail("no holding tax (gamma=1)", traceback.format_exc())

    # 11) CONTRAST: gamma<1 DOES tax held aim (the v1 failure mode) — confirms the gamma knob is wired.
    try:
        sh = ReacqShaper(ReacqShapingCfg(shaping_scale=1.0, gamma=0.99), 1, 1, device)
        b, c, t = _inputs(AIM, device=device)
        sh.compute_shaping(b, c, t, deficit_T, t=0.0)            # entry, F=0
        F1 = sh.compute_shaping(b, c, t, deficit_T, t=1.0).item()  # holding Phi=1 -> (gamma-1)*1<0
        assert abs(F1 - (0.99 * 1.0 - 1.0)) < 1e-5 and F1 < 0.0, f"expected ~-0.01 tax, got {F1}"
        results.add_pass(f"gamma<1 holding-aim tax reproduced (F={F1:.4f}<0) -> why gamma=1 is default")
    except Exception as e:
        results.add_fail("gamma<1 tax contrast", traceback.format_exc())


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    r = TestResults()
    r._emit("=" * 80)
    r._emit("REACQ_SHAPING TEST SUITE")
    r._emit("=" * 80)
    r._emit(f"Device: {device}  Torch: {torch.__version__}  Started: {datetime.now()}")
    try:
        run_tests(r, device)
    except Exception:
        r.add_error("suite", traceback.format_exc())
    ok = r.print_summary()
    _RESULT.close()
    simulation_app.close()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
