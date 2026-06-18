# Ticket 050 / Slice D — Usable peer signal: target POSITION estimate in obs

**Parent:** [../ticket.md](../ticket.md) (050 — Cooperative track re-acquisition)
**Predecessors:** [../slice-b-team-reward/](../slice-b-team-reward/) (null) ·
[../slice-c-recovery-shaping/](../slice-c-recovery-shaping/) (NO-GO + the decisive ablation)
**Status:** Stage Q (proposed). **Flow:** Full QRISPY.

## What
**Add** a resolved peer **target position estimate** (a 3D point + validity/age) to the obs, ALONGSIDE
the existing bearing `other_ray_w` (deployment review 2026-06-18: keep the reliable raw bearing, add
the uncertain point — do not replace) — so the policy gets a usable "where is the target" signal
during a single-agent deficit instead of only a bare line it cannot fuse. Re-run the A/B + the channel
ablation (now expected to bite).

## Why (what the arc proved)
- C1 (trigger) solved by Slice A; C2 (reward) addressed by Slice B (null) and Slice C shaping
  (NO-GO). **The reward axis is exhausted** — neither beat the C2 baseline (reacq 0.385).
- The Slice-C **ablation is the redirect**: masking the peer bearing *raised* reacq_success
  (0.289 → 0.311), i.e. the policy does **not use** the peer bearing. The binding constraint is
  **C3 / fusion-hardness**: a bearing is a *line*; the RNN cannot turn (peer_pos, peer_ray, ego_pos)
  into a re-point. Make the signal usable, not better-rewarded.
  (See doc/experiments/2026-06-18_ticket050_sliceC_v2_NOGO_ablation.md.)

## Approach (1 line)
Broadcast each agent's single-agent target *point* `agent_pos + range·bearing` (range = SENSOR source,
never GT — option A bbox-depth chosen) as a NEW obs channel kept alongside the bearing; the lost agent
re-aims via `normalize(peer_point − ego_pos)` — no implicit triangulation required. Reward = C2
baseline (info + shaping OFF) to isolate the obs channel. Details in i_design.md.

## Scope boundary
- IN: peer (and ego) target-point estimate obs (dim-preserving swap of the bearing slots); the
  channel ablation (mask the point); A/B vs C2; multi-seed of the headline if the single run is GO.
- OUT: reward changes (info reward + recovery shaping stay OFF — both failed); sysid envelope
  (LOCKED); scenario (Slice A reused).
- RESERVE (if the env-computed estimate is insufficient): a **learned cooperative observer**
  (recurrent belief / differentiable filter over own+peer bearings, GT-aux-supervised) — the parent's
  architecture story, escalated only on need.

## Acceptance criteria (gate)
- `reacq_success_rate` **beats C2 0.385** (and the t050b arms), under the same track_loss_scenario.
- The **ablation** (mask the peer point) is **measurably worse** than the full channel — i.e. the
  policy now *uses* it (the opposite of Slice C, where masking didn't hurt). This is the paper claim.
- No collapse of base tracking (`pair_valid_rate`) vs t048 wide.

## Build-on
t048 wide lead, warm-start. ADD changes obs_dim → **warm-start obs surgery** (append new dims at end,
zero-init new first-layer columns, expand preprocessor); one-time testable utility. Sysid LOCKED.

## Key references
- Slice C NO-GO + ablation: ../../../experiments/2026-06-18_ticket050_sliceC_v2_NOGO_ablation.md
- Per-agent depth source: bbox_raycaster_v2 `_depths` (batch_project_to_image_plane).
- Triangulation needs ≥2 cams (no single-agent fallback): triangulation/triangulation.py.
