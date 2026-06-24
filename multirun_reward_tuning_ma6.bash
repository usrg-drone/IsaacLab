#!/bin/bash
# DRAFT — iris_ma6 reward-weight tuning sweep (post-ticket-050 pivot).
# Baseline env (all ticket-050 features default-OFF). Sweeps the bbox vs triangulation balance:
# the C2 pathology was bbox(90) >> triangulation(8) (>10x selfish). Warm-start from the t048 wide
# lead and fine-tune each config (fast, apples-to-apples) — NOT from scratch.
#
# Real-run path: train_mappo_rnn_hydra.py + env.<field>=<val> hydra overrides (run_experiment.py is
# NOT the real path). Sequential on the one 3090 (concurrent training is negative-sum). 150k each.
#
# Usage:  bash multirun_reward_tuning_ma6.bash --dry_run   # print commands
#         tmux new -d -s rwtune "bash multirun_reward_tuning_ma6.bash"   # run (long; sequential)

set -u
cd /home/usrg/IsaacPX4/IsaacLab
TRAIN="./isaaclab.sh -p scripts/reinforcement_learning/skrl/train_mappo_rnn_hydra.py"
CKPT="logs/skrl/iris_ma6/2026-06-11_05-17-30_mappo_rnn_torch_ticket048_net_width_wide/checkpoints/agent_200000.pt"
TASK="Isaac-Iris-MA6-Direct-Test-v0"
TIMESTEPS=150000   # warm-start fine-tune length (~9h each)

# Grid: "tag  bbox_center  bbox_size  triangulation_scale"  (base = 90 / 30 / 8)
# 2x2 factorial: bbox_center {90,30} x triangulation {8,20}, bbox_size fixed at 30.
CONFIGS=(
  "base_confirm     90.0 30.0 8.0"    # (90, 8)  sanity: reproduce C2 reward under fine-tune
  "tri_up           90.0 30.0 20.0"   # (90,20)  boost triangulation, keep bbox
  "rebal_mid        30.0 30.0 8.0"    # (30, 8)  cut selfish bbox 90->30, triangulation unchanged
  "rebal_strong     30.0 30.0 20.0"   # (30,20)  cut bbox + boost triangulation
)

DRY=false; [[ "${1:-}" == "--dry_run" ]] && DRY=true
i=0; N=${#CONFIGS[@]}
echo "=== reward-weight tuning: $N configs, ${TIMESTEPS} steps each, warm-start $CKPT ==="
for cfg in "${CONFIGS[@]}"; do
  read -r tag bc bs tri <<< "$cfg"
  i=$((i+1))
  CMD="$TRAIN --task $TASK --headless --experiment_name rwtune_${tag} \
    --checkpoint $CKPT --timesteps $TIMESTEPS \
    env.bbox_center_reward_scale=${bc} env.bbox_size_reward_scale=${bs} env.triangulation_reward_scale=${tri}"
  echo ""; echo "[$i/$N] rwtune_${tag}: bbox_center=$bc bbox_size=$bs triangulation=$tri"
  if $DRY; then echo "  CMD: $CMD"; else
    eval "$CMD 2>&1 | tee /tmp/rwtune_${tag}.log"
    echo "[$i/$N] done rwtune_${tag} ($(date))"; sleep 5
  fi
done
echo "=== ALL DONE — compare logs/skrl/iris_ma6/*rwtune_* (triangulation reward, pair_valid_rate, trace_sigma) ==="
