# Stage P — Implementation Plan (Slice C): recovery shaping + ablation

Builds on committed Slice B (`0d965b69d9`). Default-off until the A/B confirms. No obs-dim change →
t048 warm-start loads directly.

## Phase 0 — verified anchors (done, no compute)
- ✓ Peer target bearing already in obs (`other_ray_w`, bbox-center unprojection).
- ✓ Single-agent deficit does NOT truncate (truncation needs all-blind for `tracking_lost_timeout_s`).
- ✓ `_get_states()` emits the GT-target tail from the live flag.
- ✓ Privileged-critic `__post_init__`-bypass FIXED + tested (`/tmp/test_critic_resize.py`, 6/6 PASS).

## Phase 1 — build + test (no training)
### 1a. `reacq_shaping/` module (pure-ish, stateful Φ_prev)
- `ReacqShapingCfg`: `enabled`, `shaping_scale`, `gamma=0.99`, `potential_kernel="cosine"`,
  `gate_to_deficit=True`.
- `ReacqShaper(cfg, num_envs, num_agents, device)` + `Φ_prev[N,A]`.
- `compute_shaping(bearings_w, cam_pos_w, target_pos_w, deficit_mask) -> F[N,A]` (WRITE; idempotency
  guard). `reset_idx(env_ids)`.
- `CONTEXT.md` (§4.4 Calling Contract: WRITE once/step) + `tests/run_tests.py`.
- Unit tests: Φ∈[0,1] & =1 when aimed; F ≈ γΦ′−Φ telescopes; F≈0 when already tracking; F>0 on
  re-aim toward target; gate zeros F outside DEFICIT; reset clears the spike; 2-step (unit→realistic).

### 1b. Env wiring (default-off, bit-exact when off)
- `_get_rewards`: reuse `_gather_info_signals` for bearings/cam_pos; pull `DEFICIT` mask from the
  `ReacquisitionTracker`; `rewards["reacq_shaping"] = shaping_scale * F[:,i]` (auto-sums/logs at
  [iris_ma_env6_test.py:2150](../../../../../iris_ma_env6_test.py#L2150)). `Φ_prev` reset in `_reset_idx`.
- Keep `r_diff` (Slice B) — it bought `team_track_maintenance` +0.12.

### 1c. Peer-channel ablation flag
- `peer_bearing_ablate`: zero `other_ray_w` slots in both obs paths (delayed + GT); keep
  `other_bbox_empty`. Obs dim unchanged.

### 1d. Experiments (registry)
- `t050c_shaping` (shaping ON, gt_target critic ON), `t050c_ablation` (shaping ON, `peer_bearing_ablate`),
  plus reuse committed `t050b_baseline_no_info` as control.
- Test checkpoint: short warm-start smoke logs `Episode_Reward/*_reacq_shaping` + `Coop/*`, no collapse;
  flag-off bit-exact; critic state_space sized (gt_target +3).

## Phase 2 — go/no-go (1 seed, ~15 h, sequential)
- `t050c_shaping` vs control. Sweep `shaping_scale` (2–3 values). **+ early ablation read.**
- **Decision gate:** `reacq_success_rate` clears C2 0.385 AND beats control AND ablation clearly worse.
- If null → escalate to the held-in-reserve peer position-estimate (fusion-hardness), not more reward.

## Phase 3 — rigor (multi-day, sequential; concurrent is negative-sum on the 3090)
- Multi-seed ≥3 on the winning `shaping_scale`; de-confound arms as needed (shaping-only vs
  shaping+critic); the clean single-policy channel-mask ablation figure; eval via
  `evaluate.py --track_loss_scenario --coop_metrics`.

## Gate (acceptance) — see ticket.md
reacq_success_rate & time_to_reacq beat C2 (0.385 / 0.46 s), seed-confirmed; ablation worse; base
tracking not collapsed.
