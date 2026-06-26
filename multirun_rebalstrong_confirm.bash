#!/bin/bash
# Confirm the reward-tuning winner rebal_strong (bbox_center=30, bbox_size=30, triangulation=20)
# FROM SCRATCH (no warm-start) across multiple seeds, to remove the t048-basin bias of the warm-start
# sweep. Baseline env (all ticket-050 features OFF). Full curriculum (--timesteps 400000). Sequential
# (concurrent training is negative-sum on the one 3090).
#
# Usage:  bash multirun_rebalstrong_confirm.bash --dry_run
#         tmux new -d -s rsconfirm "bash multirun_rebalstrong_confirm.bash"
# ETA: ~30h/seed @ ~3.7 it/s -> ~90h for 3 seeds. First (seed 42) result confirms the headline; the
# others add the multi-seed CI. Safe to stop after any seed.

set -u
cd /home/usrg/IsaacPX4/IsaacLab
TRAIN="./isaaclab.sh -p scripts/reinforcement_learning/skrl/train_mappo_rnn_hydra.py"
TASK="Isaac-Iris-MA6-Direct-Test-v0"
TIMESTEPS=400000          # full curriculum, from scratch
BC=30.0; BS=30.0; TRI=20.0
SEEDS=(42 123 7)

DRY=false; [[ "${1:-}" == "--dry_run" ]] && DRY=true
i=0; N=${#SEEDS[@]}
echo "=== rebal_strong from-scratch confirm: bbox=$BC/$BS tri=$TRI, ${TIMESTEPS} steps, seeds ${SEEDS[*]} ==="
for s in "${SEEDS[@]}"; do
  i=$((i+1))
  CMD="$TRAIN --task $TASK --headless --seed $s --experiment_name rsconfirm_30_30_20_seed${s} \
    --timesteps $TIMESTEPS \
    env.bbox_center_reward_scale=${BC} env.bbox_size_reward_scale=${BS} env.triangulation_reward_scale=${TRI}"
  echo ""; echo "[$i/$N] rsconfirm seed=$s (from scratch)"
  if $DRY; then echo "  CMD: $CMD"; else
    eval "$CMD 2>&1 | tee /tmp/rsconfirm_seed${s}.log"
    echo "[$i/$N] done seed $s ($(date))"; sleep 5
  fi
done
echo "=== ALL DONE — eval logs/skrl/iris_ma6/*rsconfirm_* and compare task_success vs sweep 0.602 ==="
