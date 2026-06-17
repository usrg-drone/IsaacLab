# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Potential-based recovery-shaping reward (ticket 050, Slice C).

Slice B added a team/difference information reward and came back null: during a single-agent deficit
the lost agent's cooperative reward is identically zero (bbox=0; r_diff=0 forced for no-detection), so
nothing pulls it back toward the target. This module supplies the missing **dense, per-agent gradient
on the recovery action** — re-aiming the camera at the target — via potential-based shaping.

    Phi_i(s) = 0.5 * (1 + d_i . d_i*)                # pointing alignment, in [0, 1]
        d_i  = camera boresight (world, optical axis = quat_rotate(camera_orientation_w, +z))
        d_i* = normalize(target_pos_w - cam_pos_i)   # GT bearing (privileged, reward-only)
    F_i      = scale * ( gamma * Phi_i(s') - Phi_i(s) ),   gamma = 1 by default

`d_i` is the boresight (always defined), NOT the bbox-center ray (undefined exactly when the agent has
lost the target — the regime that matters). Phi=1 when aimed at the target.

Reward = scale*(Phi' - Phi) (gamma=1), applied on steps where the agent was in a peer-assisted DEFICIT
at the START of the step. This rewards *improvement* in pointing while lost and the recovery step
itself, penalizes pointing *away*, and — crucially — does NOT tax merely holding alignment (constant
Phi -> F=0). It targets the dominant loss mode (FOV-exit -> re-aim) and is realizable: the gate is the
regime where the peer's measured bearing is in the obs, so the policy can climb it from observation. GT
enters the reward only (training-time), never the obs, so the peer-channel ablation stays valid.

v1 (gamma=0.99 + gate on the CURRENT-step deficit) failed: the -(1-gamma)*Phi drip became a net
'deficit tax' on staying aimed, and the recovery step (DEFICIT->HOLD) was gated out — regressing
tracking (doc/experiments 2026-06-17). Fixed here: gamma=1 + gate on region_prev.

Calling contract (see CONTEXT.md §4.4):
- compute_shaping(...): WRITE -- advances Phi_prev / region_prev. Call once per sim step in
                        _get_rewards; idempotent within a step via a `_last_update_time` guard.
- reset_idx(env_ids):   WRITE -- clears per-(env,agent) shaping state for the given envs.
"""

from __future__ import annotations

import torch

from .reacq_shaping_cfg import ReacqShapingCfg


class ReacqShaper:
    """Per-(env, agent) potential-based recovery shaping on pointing alignment."""

    def __init__(
        self,
        cfg: ReacqShapingCfg,
        num_envs: int,
        num_agents: int,
        device: torch.device,
    ):
        self._cfg = cfg
        self._num_envs = num_envs
        self._num_agents = num_agents
        self._device = device

        N, A, dev = num_envs, num_agents, device
        self._phi_prev = torch.zeros(N, A, device=dev)
        # Whether each (env, agent) was in the shaped region last step. Telescoping is only valid
        # across two consecutive in-region steps, so this also zeroes F on the region-entry step
        # (no spurious spike) and on the first step after a reset.
        self._region_prev = torch.zeros(N, A, dtype=torch.bool, device=dev)
        self._last_update_time = -1.0e30
        self._last_F = torch.zeros(N, A, device=dev)

    def _potential(
        self,
        boresight_w: torch.Tensor,   # [N, A, 3] camera optical axis (world)
        cam_pos_w: torch.Tensor,     # [N, A, 3] camera/agent world positions
        target_pos_w: torch.Tensor,  # [N, 3] GT target world position
    ) -> torch.Tensor:
        """Pointing potential Phi[N, A] in [0, 1] (cosine kernel)."""
        d = torch.nn.functional.normalize(boresight_w, dim=-1)
        d_star = torch.nn.functional.normalize(target_pos_w.unsqueeze(1) - cam_pos_w, dim=-1)
        cos = (d * d_star).sum(dim=-1)  # [N, A] in [-1, 1]
        return 0.5 * (1.0 + cos)

    def compute_shaping(
        self,
        boresight_w: torch.Tensor,   # [N, A, 3]
        cam_pos_w: torch.Tensor,     # [N, A, 3]
        target_pos_w: torch.Tensor,  # [N, 3]
        deficit_mask: torch.Tensor,  # [N, A] bool — agent is in a peer-assisted DEFICIT
        t: float,
    ) -> torch.Tensor:
        """Per-agent shaping reward F[N, A] = scale * (gamma*Phi' - Phi), gated to the deficit regime.

        WRITE — call exactly once per sim step. Idempotent within a step (returns the cached F).
        """
        t = float(t)
        if abs(t - self._last_update_time) < 1e-6:
            return self._last_F
        self._last_update_time = t

        phi = self._potential(boresight_w, cam_pos_w, target_pos_w)  # [N, A]

        # Shaped region: the deficit regime (realizable) or everywhere (pure PBRS ablation).
        if self._cfg.gate_to_deficit:
            region = deficit_mask.bool()
        else:
            region = torch.ones_like(deficit_mask, dtype=torch.bool)

        # Gate on whether the agent was in the region at the START of this step (region_prev), NOT
        # `region & region_prev`. This credits: entry step -> 0 (region_prev False), mid-deficit
        # re-pointing, AND the recovery step (DEFICIT->HOLD: region False but region_prev True, so the
        # big +ΔΦ that achieves recovery is rewarded — v1 gated it out). First post-reset step -> 0.
        active = self._region_prev
        f_raw = self._cfg.gamma * phi - self._phi_prev  # gamma=1 -> pure ΔΦ, no holding tax [N, A]
        F = torch.where(active, f_raw, torch.zeros_like(f_raw)) * self._cfg.shaping_scale

        # Advance state — copy INTO the owned buffers. Do NOT do `self._region_prev = region`:
        # `deficit_mask.bool()` aliases the caller's tensor when it is already bool, and reset_idx
        # mutates `_region_prev` in place, which would corrupt the caller's deficit mask.
        self._phi_prev.copy_(phi)
        self._region_prev.copy_(region)
        self._last_F.copy_(F)
        return F

    def reset_idx(self, env_ids: torch.Tensor | None = None) -> None:
        """Clear shaping state for ``env_ids`` (all envs if None). WRITE."""
        if env_ids is None:
            env_ids = slice(None)
        self._phi_prev[env_ids] = 0.0
        self._region_prev[env_ids] = False
        self._last_F[env_ids] = 0.0
