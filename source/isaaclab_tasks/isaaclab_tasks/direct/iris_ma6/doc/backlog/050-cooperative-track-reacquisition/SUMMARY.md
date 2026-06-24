# Ticket 050 — Cooperative track re-acquisition: PARKED (backlog, future research)

**Status:** Parked 2026-06-19. Moved from `doc/active/ticket/` to backlog. The env is reverted to
baseline (all 050 features default-OFF, verified bit-exact); focus shifted to reward-weight tuning.

## Goal (the intended journal contribution)
Agents re-acquire a lost target using **peer information** under perception loss (FOV exit / dropout);
central claim = an ablation where removing the peer channel breaks re-acquisition.
Baseline to beat: **C2 reacq_success = 0.385** (Slice A, on the t048 wide policy).

## What we tried and what happened (deterministic eval, 256 envs, step-400k)
| arm | reacq_succ | note |
|---|---|---|
| C2 baseline | 0.385 | competent tracker, no cooperation |
| Slice B — team/difference info reward | 0.359 | **NULL** (lost agent gets 0 reward gradient in deficit) |
| Slice C — gated recovery shaping (PBRS) | 0.289 | **NO-GO** (v1 deficit-tax; v2 fixed but still a drag) |
| Slice D — peer target POSITION estimate (oracle point ≈ GT) | 0.398 | **MARGINAL** (+0.013, within noise) |
| Slice D ablation (point masked) | 0.377 | channel barely used (~1 SE) |

## Conclusion — why it's parked
Two axes eliminated by experiment:
- **Reward/incentive** — Slice B null, Slice C no-go.
- **Information/fusion** — Slice D: handed an **oracle target point (≈ GT position)**, reacq only ~0.40
  and removing it barely matters (ablation flat). Knowing *where* is not the blocker.

By elimination the binding constraint is the **control envelope / scenario recoverability**: after a
FOV-exit the agent likely can't slew/reposition fast enough to re-acquire within the episode, *even
knowing where the target is* — consistent with t045/t048 slew saturation and the sysid-lock.

## How to resume (the parked substrate — code kept, default-off)
Modules (all gated off; flip the cfg flag to re-activate):
- `cooperation_metrics/` (ReacquisitionTracker — deficit/reacq instrumentation) — `cooperation_metrics.enable`
- `information_reward/` (FIM difference reward) — `information_reward.enabled`
- `reacq_shaping/` (PBRS recovery shaping, γ=1 v2) — `reacq_shaping.enabled`
- `track_loss_scenario_cfg.py` (scenario ceiling) — `enable_track_loss_scenario`
- peer target estimate obs + ablation — `peer_target_estimate` / `peer_target_estimate_ablate` / `peer_bearing_ablate`
- privileged critic — `enable_critic_gt_target` (sizing fix is in `finalize_observation_and_state_spaces`)
- Experiments `t050b_*`, `t050c_*`, `t050d_*` in `experiments/experiment_registry.py`.
- Per-run analysis in `doc/experiments/2026-06-1{6,7,8,9}_ticket050_*`.

**Recommended FIRST step if resumed:** the **control-envelope diagnostic** (analysis only) — slew
saturation during deficits; fraction of structurally-recoverable deficits (reachable-set-within-FOV
gate `v_max·bbox_age` vs `k·range·tan(fov_half)`); reacq conditioned on recovered/not. Only after that:
calibrate the Slice-A scenario to the locked envelope, re-frame the contribution, or revisit sysid-lock.
Also: Slice D used **oracle range** (option A); the deploy-faithful **EKF bearing-only** (option D) is
the realistic estimator — but it would land ≤ 0.398, so it does not change the conclusion.

Commits: `b4edcfa5af` (Slice A) → `18f0a2bb70` (Slice D verdict).
