# iris_ma6 reward-weight tuning — 2×2 bbox×triangulation sweep

**Date**: 2026-06-27
**Setup**: warm-start fine-tune from t048 wide (`agent_200000.pt`), **baseline env** (all ticket-050
features OFF), 150k steps each, single seed (42), sequential. 2×2 factorial over
`bbox_center_reward_scale ∈ {90,30}` × `triangulation_reward_scale ∈ {8,20}`, `bbox_size=30` fixed.
Script: `multirun_reward_tuning_ma6.bash`. Eval: `experiments/outputs/rwtune/*.json` (~1025 eps, SE≈0.016).

## Eval — FINAL checkpoints (same-budget comparison)
| config | bbox/size/tri | task_succ ↑ | tri_rmse ↓ | traceΣ ↓ | track_maint ↑ | collision ↓ |
|---|---|---|---|---|---|---|
| base_confirm | 90/30/8 | 0.468 | 2.24 | 63.6 | 0.814 | 1.46 |
| tri_up | 90/30/20 | 0.497 | 1.86 | 130.8 | 0.701 | 1.51 |
| rebal_mid | 30/30/8 | 0.550 | 1.84 | 102.4 | 0.818 | 1.77 |
| **rebal_strong** | **30/30/20** | **0.602** | **1.62** | 109.6 | 0.797 | **1.25** |

(best-checkpoint task_succ: base 0.550, tri_up 0.548, rebal_mid 0.536, rebal_strong 0.602 — only
rebal_strong is stable, best==final; base *degraded* best 0.550 → final 0.468 over the fine-tune.)

## Findings
1. **`rebal_strong` (30/30/20) is the clear winner** — task_success 0.602 vs base 0.468 (+0.134, ~8 SE),
   best triangulation RMSE (1.62, ~28% lower), lowest collisions (1.25), and stable (best==final).
2. **2×2, both axes help, ~additive:**
   - **Cutting bbox_center 90→30 is the big lever** (+0.08–0.10 task_succ) — removes the C2 "fly close
     to fill my own frame" selfish pathology.
   - **Boosting triangulation 8→20** adds +0.03–0.05, mostly once the bbox cut is in. `tri_up` alone
     (boost tri, keep bbox 90) is the weakest rebalance (even hurt track_maint / pair_valid).
3. **`traceΣ` caveat:** rebalanced configs report *higher* covariance trace (109.6) than base (63.6)
   despite *lower* actual RMSE — likely a standoff/range artifact (base flies close → small covariance
   but overconfident/high error; rebalanced stands off → larger covariance ∝ range² but better geometry
   → lower RMSE). Confirm via `trace_sigma_median` + range correlation; RMSE (actual error) favors
   rebalanced, so estimation is genuinely better.
4. TB corroboration: weights applied (bbox_rew 57→16 as 90→30; tri_rew 51→143 as 8→20); training
   healthy (σ 1.23–1.26, tracking_lost_fraction 0.02–0.05).

## From-scratch confirmation (2026-07-05) — the warm-start win did NOT transfer
`rebal_strong` (30/30/20) trained FROM SCRATCH, 400k, seeds 42/123/7 (`multirun_rebalstrong_confirm.bash`),
evaluated identically (1024 envs, baseline). **task_success = 0.418 ± 0.027** — far below the warm-start
0.602, and *below the warm-start base 0.468*. All quality metrics worse (tri_rmse 2.28, tri_valid 0.60,
track_maint 0.59). Tight across seeds → robust, not a bad seed.

**Conclusion: the sweep's 0.602 was warm-start-DEPENDENT, not a property of the weights.** bbox_center=90
is a dense, easy-to-learn bootstrap for basic tracking; warm-start from t048 supplied a bbox-90-trained
tracker, so fine-tuning to 30/30/20 kept tracking and added cooperative geometry. From scratch, cutting
bbox to 30 at step 0 starves the tracking bootstrap → the policy under-learns tracking → everything worse.
(Training-time pair_valid looked fine ~0.85 stochastic; the deterministic full-difficulty eval exposes it.)

## Decision / next — do NOT adopt constant 30/30/20 from scratch
The rebalanced reward helps only WITH a bootstrap. Two ways to capture it:
- **(a) Warm-start / fine-tune pipeline** (proven 0.602): train a bbox-90 tracker, then rebalance-fine-tune.
- **(b) bbox-rebalance CURRICULUM** (bbox 90→30 over training) in one from-scratch run — bootstrap early,
  rebalance late. This is the Slice-B `reward_rebalance` machinery, currently gated behind
  `information_reward.enabled`; decouple the bbox-schedule part (small change) and re-confirm from scratch.
- Still open: `trace_sigma` median/range check (standoff hypothesis) on the warm-start runs.

Artifacts: `experiments/outputs/rwtune/{...,rsconfirm_seed{42,123,7}}.json`, runs `*rwtune_*` / `*rsconfirm_*`.

Artifacts: `experiments/outputs/rwtune/{base_confirm,tri_up,rebal_mid,rebal_strong}{,_final}.json`,
runs `logs/skrl/iris_ma6/*rwtune_*`, sweep commit `be85ff2753`.
