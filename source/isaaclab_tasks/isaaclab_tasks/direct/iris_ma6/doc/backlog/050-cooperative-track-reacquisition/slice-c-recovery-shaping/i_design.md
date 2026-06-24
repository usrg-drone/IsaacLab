# Stage I — Design (Slice C): gated recovery shaping + peer-channel ablation

## Problem
During a single-agent deficit the lost agent gets **zero cooperative reward gradient** (bbox=0;
`r_diff=0` forced for no-detection). The peer's target bearing is already observable (`other_ray_w`),
but nothing *pays* the lost agent to act on it. Slice B (reward form alone) was null. The missing
piece is a **dense, realizable incentive to re-point during the deficit**.

## Approach — potential-based recovery shaping (PBRS on pointing)
Per agent i, pointing potential on the **GT bearing** (privileged, reward-only — same class as the
existing `r_diff`/GT-range):

    d_i  = agent i camera ray (world)          # = other obs already exposes this as a measurement
    d_i* = normalize(target_pos_w − cam_pos_i)  # GT bearing (privileged)
    Φ_i  = ½ (1 + d_i · d_i*)        ∈ [0, 1]    # 1 when aimed at the target

Shaping reward (Ng et al. 1999, optimum-preserving):

    F_i = γ · Φ_i(s′) − Φ_i(s),   γ = 0.99 (agent discount_factor)

added as a NEW reward key `reacq_shaping` scaled by `shaping_scale`, **gated to the DEFICIT regime**
(ego-lost ∧ peer-holds ∧ reachable — the `ReacquisitionTracker` state).

**Why this and not Slice-B variants:**
- *Optimum-preserving*: PBRS densifies the recovery PATH without biasing the task optimum (which
  already rewards re-acquisition via bbox + `r_diff` once achieved). Fixes the sparse/delayed credit.
- *Targets the real loss mode*: eval shows `event_rate_fov ≈` all losses → recovery = **re-aim**, not
  translate. (Slice B chased `reacq_dist_delta` "close-in", which went the wrong way.)
- *Realizable*: gated to where `other_ray_w` is in the obs (peer holds), so the policy can actually
  climb the reward from observation. C2's `reacq_bearing_align=0.80` confirms agents DO use the peer
  ray when they recover — the gap is the *rate* of attempts, i.e. the incentive. (See realizability
  principle in ticket.md.)

**Ablation validity:** Φ uses GT only in the *reward* (training-time). The policy never observes GT;
at deploy it must infer the re-point direction from obs — whose only target-direction signal during
its own blackout is the peer's `other_ray_w` + `other_bbox_empty`. So masking that channel still
breaks recovery. Privileged shaping ≡ privileged critic — standard, no leak.

## Statefulness (Calling Contract)
`Φ_prev[N,A]` buffer. `compute_shaping(...)` is a **WRITE** method (advances `Φ_prev`) — call once per
step in `_get_rewards`; apply the `_last_update_time` idempotency guard (CLAUDE.md Stateful Rules).
`reset_idx(env_ids)` re-seeds `Φ_prev` to the current Φ (no spurious ΔΦ spike on the first post-reset
step). On truncation (time-out / all-blind), standard bootstrap — PBRS telescoping is unaffected
because single-agent deficits do NOT truncate ([iris_ma_env6_test.py:2931](../../../../../iris_ma_env6_test.py#L2931)).

## Peer-channel ablation
`peer_bearing.ablate` flag that **zeros `other_ray_w`** in the peer block (keep `other_bbox_empty` so
the agent still knows a recoverable deficit exists, isolating the *bearing's* value). Obs dim fixed →
a single trained policy can be evaluated channel-on vs channel-off (cleanest figure), and a
trained-without arm gives the stronger claim.

## Privileged critic (now enabled — the `__post_init__`-bypass fix)
`enable_critic_gt_target=True` for the shaping runs (GT target → critic only; actor obs unchanged),
stabilizing V through the deficit. The Hydra-bypass that blocked this in Slice B is **fixed**:
`cfg.finalize_observation_and_state_spaces()` is called at env `__init__` pre-super() so an
override-set flag re-sizes `state_space` (verified: +3 for gt_target, +6 with velocity, idempotent).

## Held in reserve (only if Phase 2 plateaus AND diagnostics show fusion-hardness)
A peer target **position estimate** in obs (ray resolved to a point via range prior / peer's own
single-agent estimate) — offloads the parallax fusion. Distinct from the redundant bearing.

## Open risks
1. `shaping_scale` sets the recovery pull; calibrate so a full recovery's shaping return ≈ the bbox
   reward gained on re-acquisition. Sweep 2–3 values in Phase 2.
2. Gating relaxes strict PBRS optimality-invariance at regime boundaries (bounded; acceptable —
   realizability > purity).
3. Φ kernel (cosine vs sharper) — cosine default; finalize in S.
