# Ticket 050 / Slice C — v2 recovery-shaping: deterministic eval = NO-GO; ablation redirects the arc

**Date**: 2026-06-18
**Run**: `2026-06-17_11-11-49_mappo_rnn_torch_t050c_shaping` (200k, 17.5h), v2 fix
(γ=1, gate on region_prev, shaping_scale=10), warm-start t048 wide. = `t050b_team_reward` + shaping.
**Eval**: `evaluate.py --track_loss_scenario --step 400000 --num_envs 256`, seed 42 (auto-eval queue).

## Result — NO-GO
| arm | reacq_succ | t_reacq_s | team_maint | trackloss |
|---|---|---|---|---|
| C2 baseline (Slice A) | **0.385** | 0.460 | 0.530 | 2.05 |
| t050b_baseline_no_info | 0.349 | 0.573 | 0.445 | 1.53 |
| t050b_team_reward | 0.359 | 0.735 | 0.566 | 1.43 |
| **t050c_shaping (v2)** | **0.289** | 0.552 | 0.434 | 1.67 |
| **t050c ABLATION (peer bearing masked)** | **0.311** | 0.511 | 0.454 | 1.82 |

1. **Shaping made re-acquisition worse.** t050c reacq_success 0.289 < C2 0.385 < ... < everything.
   The −0.10 vs C2 is well beyond noise (binomial SE ≈0.02). v2 prevented the v1 *collapse* (no hard
   tracking degradation mid-run) but the shaping is still a net drag; `reacq_shaping` stayed net
   ~−0.04 the whole run (failed deficits, where Φ decays as the target moves, outweigh credited
   recoveries). Reward-only axis = clean NO-GO.
2. **The ablation is backwards — the policy does NOT use the peer bearing.** Masking `other_ray_w`
   *raised* reacq_success (0.289 → 0.311). The central claim needs masking to make it WORSE; it's
   flat/slightly-better (within noise). The trained policy re-acquires from its own recurrent
   belief/search, not from peer info.

## What this means for the arc
- **Reward axis exhausted.** Slice B (info/difference reward) → null; Slice C (PBRS recovery shaping,
  even after the v1 deficit-tax fix) → negative. Neither beat C2 on re-acquisition.
- **The ablation localizes the bottleneck.** It is NOT under-incentivization (reward) — it is that the
  cooperative channel is *unused*. A raw peer bearing *ray* (a line) is too hard for the RNN to fuse
  into a re-point (C3 / fusion-hardness), so no reward shaping on top can teach recovery-via-peer.

## Decision / next
Promote the RESERVE to primary: give the peer's target as a **resolved position estimate** in obs
(peer ray × peer range / peer single-agent triangulation → a 3D point + uncertainty), and/or a
**learned cooperative observer** (recurrent belief / differentiable filter over own+peer bearings).
Re-run the same A/B + the channel ablation (now expected to bite, if the estimate is usable). Keep the
shaping OFF (or revisit only once the channel is usable). Sysid envelope still LOCKED.

Artifacts: /tmp/t050c_eval_{shaping,ablation}.json, /tmp/t050c_eval.log. Predecessor (v1 deficit tax):
doc/experiments/2026-06-17_ticket050_sliceC_v1_deficit_tax.md.
