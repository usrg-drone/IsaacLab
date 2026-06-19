# Ticket 050 / Slice D — peer position estimate (oracle point): eval = MARGINAL; redirects to control envelope

**Date**: 2026-06-19
**Run**: `2026-06-18_12-07-48_mappo_rnn_torch_t050d_posest` — CLEAN START from scratch, 400k (31h),
`peer_target_estimate=True` (range = option A, sim true range ⇒ point ≈ GT target), reward axis OFF
(info + shaping off), scenario + coop on.
**Eval**: `evaluate.py --track_loss_scenario --step 400000 --num_envs 256 --peer_target_estimate`
(+ `--peer_target_estimate_ablate` for the ablation).

## Result — marginal / inconclusive (gate NOT cleanly met)
| arm | reacq_succ | t_reacq_s | team_maint | trackloss |
|---|---|---|---|---|
| C2 baseline | 0.385 | 0.460 | 0.530 | 2.05 |
| t050b_team | 0.359 | 0.735 | 0.566 | 1.43 |
| t050c_shaping | 0.289 | 0.552 | 0.434 | 1.67 |
| **t050d_posest (oracle point)** | **0.398** | 0.516 | 0.523 | 1.75 |
| **t050d ABLATION (point masked)** | **0.377** | 0.572 | 0.488 | 1.67 |

- 0.398 vs C2 0.385 = +0.013, within noise (SE ≈ 0.016 over ~900 qualified deficits). Nominally the
  best run, but not a statistically clear win.
- Ablation 0.398→0.377 (−0.021, ~1 SE): the channel is at most weakly used; masking does not clearly
  degrade re-acquisition (the paper claim does NOT bite).
- (Training-time stochastic reacq was ~0.45 — the usual overstatement vs deterministic eval.)

## The finding — information is NOT the binding constraint
We handed the policy an **oracle target point (≈ GT target position)** and re-acquisition still only
reached ~0.40, with the channel barely used. Across the arc:
- Reward axis: Slice B (info/difference reward) null; Slice C (shaping) NO-GO.
- Information axis: Slice D (oracle target point) marginal, ablation flat.

Elimination ⇒ the bottleneck is **neither incentive nor information/fusion**. The remaining axis is the
**control envelope / scenario recoverability**: after a FOV-exit (the dominant loss mode), the agent
likely cannot slew the gimbal / reposition fast enough to re-acquire within the episode — *regardless
of knowing where the target is*. Consistent with the t045/t048 **slew-saturation** result and the
**sysid-lock** (envelope held fixed by design).

## Decision / next — DIAGNOSE before more training
Before any further run, quantify WHY deficits don't recover even with the oracle point:
- Slew saturation during deficits (are the gimbal/yaw channels pinned at the rate limit while trying to
  re-aim?). t045 already showed ~100% slew saturation.
- Fraction of deficits that are *structurally recoverable* given the locked envelope: does the
  reachable-set-within-FOV gate (v_max·bbox_age vs k·range·tan(fov_half)) say the target is even
  re-aimable before the episode ends?
- reacq_dist_delta / time-to-reacq conditioned on recovered vs not.

If control-bound: the Slice-A scenario ceiling may be too aggressive for the locked envelope. Options
then are (a) calibrate the scenario so deficits are recoverable within the envelope, (b) re-frame the
contribution, or (c) revisit the sysid-lock — explicitly out of scope without this diagnosis.
NOTE: range=A is oracle; the deploy-faithful EKF (option D) would land at/below 0.398, so it does not
change this conclusion.

Artifacts: /tmp/t050d_eval_{posest,ablation}.json, /tmp/t050d_eval.log.
