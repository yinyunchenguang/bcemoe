#!/usr/bin/env bash
set -u

# Sequential runner:
# 1) skip YAGO3-10 seed 21
# 2) make sure original is fully finished: model_final.pt after 2000 epochs
# 3) then run gnndelete_nodeemb scgu_moe

ROOT="/root/autodl-tmp/GNNDelete-main1/GNNDelete-main"
cd "$ROOT" || exit 1

# Avoid wandb login/API-key failures in nohup/background jobs.
export WANDB_MODE=offline
export WANDB_SILENT=true

mkdir -p logs/scgu_moe_remaining_multiseed

DATASETS=(YAGO3-10 codex-l codex-m WordNet18RR)
GNNS=(rgcn rgat)
SEEDS=(13 21 42 87 100)

UNLEARNING_MODEL="gnndelete_nodeemb"
DF="random"
DF_SIZE="0.5"
ALPHA="0.5"

skip_combo() {
  local dataset="$1"
  local gnn="$2"
  local seed="$3"

  # User requested: skip yago3 seed21.
  if [[ "$dataset" == "YAGO3-10" && "$seed" == "21" ]]; then
    return 0
  fi
  return 1
}

original_done() {
  local dataset="$1"
  local gnn="$2"
  local seed="$3"
  [[ -f "checkpoint/${dataset}/${gnn}/original/${seed}/model_final.pt" ]]
}

delete_done() {
  local dataset="$1"
  local gnn="$2"
  local seed="$3"
  local outdir="checkpoint/${dataset}/${gnn}/${UNLEARNING_MODEL}/mse_mean-both_layerwise-${ALPHA}-non_connected-bce_rank_l2-0.0-1.0-scgu_moe-16-random-4-16-soft/${DF}-${DF_SIZE}-${seed}"
  [[ -f "${outdir}/model_final.pt" || -f "${outdir}/model_best.pt" ]]
}

original_running() {
  local dataset="$1"
  local gnn="$2"
  local seed="$3"
  pgrep -f "python train_gnn.py --unlearning_model original --gnn ${gnn} --dataset ${dataset} --random_seed ${seed}" >/dev/null
}

wait_original_done() {
  local dataset="$1"
  local gnn="$2"
  local seed="$3"
  local log="$4"

  while original_running "$dataset" "$gnn" "$seed"; do
    if original_done "$dataset" "$gnn" "$seed"; then
      return 0
    fi
    echo "[WAIT original running] $(date '+%F %T') $dataset $gnn seed=$seed : waiting for checkpoint/.../model_final.pt" | tee -a "$log"
    sleep 300
  done

  original_done "$dataset" "$gnn" "$seed"
}

for dataset in "${DATASETS[@]}"; do
  for gnn in "${GNNS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      log="logs/scgu_moe_remaining_multiseed/${dataset}_${gnn}_${DF}${DF_SIZE}_seed${seed}.log"

      if skip_combo "$dataset" "$gnn" "$seed"; then
        echo "[SKIP requested] $dataset $gnn seed=$seed" | tee -a "$log"
        continue
      fi

      if delete_done "$dataset" "$gnn" "$seed"; then
        echo "[SKIP delete done] $dataset $gnn seed=$seed" | tee -a "$log"
        continue
      fi

      if [[ ! -f "data/${dataset}/d_${seed}.pkl" || ! -f "data/${dataset}/df_${seed}.pt" ]]; then
        echo "[SKIP no split] $dataset $gnn seed=$seed : missing data/${dataset}/d_${seed}.pkl or df_${seed}.pt" | tee -a "$log"
        continue
      fi

      # Important: original must be complete, not just directory exists.
      if ! original_done "$dataset" "$gnn" "$seed"; then
        if original_running "$dataset" "$gnn" "$seed"; then
          wait_original_done "$dataset" "$gnn" "$seed" "$log"
        fi
      fi

      if ! original_done "$dataset" "$gnn" "$seed"; then
        echo "============================================================" | tee -a "$log"
        echo "[START original] $(date '+%F %T') dataset=$dataset gnn=$gnn seed=$seed epochs=2000" | tee -a "$log"
        echo "============================================================" | tee -a "$log"

        python train_gnn.py \
          --unlearning_model original \
          --gnn "$gnn" \
          --dataset "$dataset" \
          --random_seed "$seed" \
          --epochs 2000 \
          2>&1 | tee -a "$log"

        status=${PIPESTATUS[0]}
        echo "[END original] $(date '+%F %T') dataset=$dataset gnn=$gnn seed=$seed status=$status" | tee -a "$log"

        if [[ "$status" -ne 0 || ! -f "checkpoint/${dataset}/${gnn}/original/${seed}/model_final.pt" ]]; then
          echo "[SKIP delete] original not complete: $dataset $gnn seed=$seed" | tee -a "$log"
          continue
        fi
      fi

      echo "============================================================" | tee -a "$log"
      echo "[START delete] $(date '+%F %T') dataset=$dataset gnn=$gnn seed=$seed" | tee -a "$log"
      echo "============================================================" | tee -a "$log"

      python delete_gnn.py \
        --unlearning_model "$UNLEARNING_MODEL" \
        --gnn "$gnn" \
        --dataset "$dataset" \
        --deletion_operator scgu_moe \
        --loss_r_type bce_rank_l2 \
        --scgu_rank 16 \
        --scgu_init random \
        --del_moe_num_experts 4 \
        --del_gate_emb_dim 16 \
        --del_gate_mode soft \
        --del_gate_temperature 1.0 \
        --alpha "$ALPHA" \
        --loss_r_margin 0.0 \
        --loss_r_beta 1.0 \
        --df "$DF" \
        --df_size "$DF_SIZE" \
        --random_seed "$seed" \
        2>&1 | tee -a "$log"

      status=${PIPESTATUS[0]}
      echo "[END delete] $(date '+%F %T') dataset=$dataset gnn=$gnn seed=$seed status=$status" | tee -a "$log"

      if [[ "$status" -ne 0 ]]; then
        echo "[WARN] delete failed: $dataset $gnn seed=$seed, continue next one" | tee -a "$log"
      fi
    done
  done
done

echo "[ALL DONE] $(date '+%F %T')"
