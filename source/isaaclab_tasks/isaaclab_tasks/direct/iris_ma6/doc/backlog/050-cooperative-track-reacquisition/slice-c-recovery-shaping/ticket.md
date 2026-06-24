# Ticket 050 / Slice C — Recovery-path shaping + peer-channel ablation

**Parent:** [../ticket.md](../ticket.md) (050 — Cooperative track re-acquisition)
**Predecessor:** [../slice-b-team-reward/](../slice-b-team-reward/) (DONE; **null result** — see
[doc/experiments/2026-06-16_ticket050_sliceB_team_reward_AB.md](../../../../experiments/2026-06-16_ticket050_sliceB_team_reward_AB.md))
**Status:** Plan locked (Stage P). Build = Phase 1.
**Flow:** Full QRISPY.

## What
Make a lost agent **re-acquire using the peer bearing**, by adding a **dense, gated recovery reward**
for re-pointing during a single-agent deficit — and run the **peer-channel ablation** that is the
paper's central claim. No new obs channel is needed (see the correction below); the ablation operates
on the existing `other_ray_w`.

## Why (what Slice B taught us)
Slice B added the FIM team/difference reward and came back **null**: `reacq_success_rate` 0.359 (team)
vs 0.349 (no-info control) vs 0.385 (C2) — no improvement, and `time_to_reacq` got worse (0.735 s).
Root cause: during a single-agent deficit the lost agent's reward is **identically zero** for every
cooperative term (bbox=0, and `r_diff=0` is forced for an agent without a valid detection) — the C2
sparse/delayed-credit pathology, relocated. Nothing pulls it back toward the target.

## Correction to the parent diagnosis (C3) — VERIFIED 2026-06-16
Parent-ticket C3 claims "agent receives peer **gimbal pointing**, not peer target bearing/estimate."
**This is stale.** The peer obs block carries `other_ray_w = camera_ray_directions_w[:,0,:]`
([iris_ma_env6_test.py:2424,2503](../../../../../iris_ma_env6_test.py#L2424)), computed by
`compute_ray_directions_from_bbox` ([delay_system_v3/derived_field_computers.py:140](../../../../../delay_system_v3/derived_field_computers.py#L140)) — it **unprojects the 2D bbox center through the
zoom-adjusted intrinsics to a world-frame bearing**. So the peer's *measured target bearing* is
already broadcast, gated by `other_bbox_empty` and tagged with `bbox_age`. The peer block contains no
gimbal joint angles at all. → "Slice C = add peer bearing to obs" is **redundant**; the lever is the
**reward gradient**, and the ablation runs on the existing channel.

## Realizability principle (the design constraint)
A shaping reward must have an **observable predictor present whenever it fires**. Clean-bbox reward
passes (noisy bbox always co-present in obs). A GT-bearing reward applied *everywhere* fails in the
both-blind regime (no predictor). → **gate the recovery shaping to the single-agent-deficit regime**
(ego-lost ∧ peer-holds ∧ reachable, the `ReacquisitionTracker.DEFICIT` state), where the peer's
`other_ray_w` is in the obs. Then `GT bearing : other_ray_w in obs :: clean bbox : noisy bbox in obs`.

## Scope boundary
- IN: gated recovery shaping (PBRS on pointing, GT bearing, reward-only/privileged); peer-channel
  ablation (mask `other_ray_w`); restore the privileged critic (`enable_critic_gt_target`, now safe
  via the `__post_init__`-bypass fix); multi-seed (≥3) of the headline.
- OUT: adding a peer *bearing* to obs (already present); architecture (Slice D); sysid envelope
  (LOCKED); the scenario (Slice A, reused). **HELD IN RESERVE:** a peer *position estimate* in obs
  (ray resolved to a point) — only if Phase 2 plateaus and diagnostics show fusion-hardness, not an
  incentive gap.

## Acceptance criteria (gate)
- `reacq_success_rate` & `time_to_reacq` **improve vs the C2 baseline** (0.385 / 0.46 s),
  seed-confirmed (≥3), under the same `track_loss_scenario`.
- The **ablation** (mask `other_ray_w`, shaping trained-with) is **measurably worse** — the paper claim.
- No collapse of base tracking (`pair_valid_rate`, triangulation) vs the t048 wide baseline.

## Build-on
t048 wide lead (warm-start, **direct load — no obs-dim change**) + `enable_track_loss_scenario` +
`cooperation_metrics.enable`. Sysid LOCKED.

## Key references
- Slice B null + root cause: [doc/experiments/2026-06-16_ticket050_sliceB_team_reward_AB.md](../../../../experiments/2026-06-16_ticket050_sliceB_team_reward_AB.md)
- C2 baseline: [../slice-a-scenario-trigger/eval_baseline_results.md](../slice-a-scenario-trigger/eval_baseline_results.md)
- Privileged-critic `__post_init__` fix: `cfg.finalize_observation_and_state_spaces()` +
  env `__init__` pre-super() call (verified `/tmp/test_critic_resize.py`).
