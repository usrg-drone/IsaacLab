# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the gated recovery-shaping reward (ticket 050, Slice C)."""

from __future__ import annotations

from isaaclab.utils import configclass


@configclass
class ReacqShapingCfg:
    """Configuration for :class:`ReacqShaper`.

    A potential-based shaping reward (Ng et al. 1999) on the agent's *pointing alignment* to the
    target, **gated to the single-agent-deficit regime** so it is realizable from observation (the
    peer's measured bearing ``other_ray_w`` is in the obs exactly when a peer holds the target). It
    densifies the otherwise sparse/delayed credit for re-pointing during a deficit — the gap Slice B
    left. Reward-side only; uses the privileged GT target (never enters obs).
    """

    enabled: bool = False
    """Whether the recovery shaping is active. Default False = bit-exact baseline."""

    shaping_scale: float = 2.0
    """Scale on the per-step potential difference F = scale * (gamma*Phi' - Phi). Calibrate so a full
    recovery's shaping return is comparable to the bbox reward gained on re-acquisition (Phase-2 sweep).
    NOTE: this is a per-step PBRS term, NOT a rate*dt term — it is added to the reward without the
    step_dt factor the other reward components carry."""

    gamma: float = 0.99
    """Discount used in the potential difference. MUST match the agent's discount_factor for the PBRS
    optimality-invariance to hold (skrl_mappo_cfg.yaml discount_factor=0.99)."""

    gate_to_deficit: bool = True
    """If True (default), F is nonzero only while the agent is in a peer-assisted DEFICIT
    (ReacquisitionTracker state) — the realizable regime. If False, F applies in every state (pure
    PBRS); requires nothing from the tracker (an ablation arm)."""

    potential_kernel: str = "cosine"
    """Pointing potential kernel. "cosine": Phi = 0.5*(1 + d . d*), d = camera boresight (world),
    d* = GT bearing to target. Only "cosine" is implemented; field reserved for future kernels."""
