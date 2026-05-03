#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/vepfs-mlp2/c20250518/241506050/code/video_code/data_construct"
PROMPT_BANK_DIR="${PROMPT_BANK_DIR:-${ROOT_DIR}/artifacts/prompt_banks/rf_prompt_bank_v1}"
DATASET_ROOT="${DATASET_ROOT:-${ROOT_DIR}/artifacts/datasets/wan13b_teacher_v1_train}"
MODEL_ROOT="${MODEL_ROOT:-/vepfs-mlp2/c20250518/241506050/ckpts/wan21/Wan2.1-T2V-1.3B}"
WAN_PYTHON="${WAN_PYTHON:-/vepfs-mlp2/c20250518/241506050/miniconda3/envs/wan/bin/python}"

TRAIN_COUNT="${TRAIN_COUNT:-20000}"
VAL_COUNT="${VAL_COUNT:-1000}"
PROMPT_SPLIT="${PROMPT_SPLIT:-train}"
MAX_ITEMS="${MAX_ITEMS:-}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"

mkdir -p "${PROMPT_BANK_DIR}"
mkdir -p "${DATASET_ROOT}"

"${WAN_PYTHON}" "${ROOT_DIR}/scripts/build_prompt_bank.py" \
  --output-dir "${PROMPT_BANK_DIR}" \
  --train-count "${TRAIN_COUNT}" \
  --val-count "${VAL_COUNT}"

PROMPT_BANK_PATH="${PROMPT_BANK_DIR}/${PROMPT_SPLIT}.jsonl"

RENDER_ARGS=(
  "${ROOT_DIR}/scripts/render_teacher_dataset.py"
  --prompt-bank "${PROMPT_BANK_PATH}"
  --output-dir "${DATASET_ROOT}"
  --model-root "${MODEL_ROOT}"
  --output-ext mp4
  --batch-size "${BATCH_SIZE}"
)

if [[ -n "${MAX_ITEMS}" ]]; then
  RENDER_ARGS+=(--max-items "${MAX_ITEMS}")
fi

if [[ "${WORKERS_PER_GPU}" -le 1 ]]; then
  TORCHRUN_CMD=(
    "${WAN_PYTHON}"
    -m
    torch.distributed.run
    --standalone
    --nproc_per_node="${NPROC_PER_NODE}"
    "${RENDER_ARGS[@]}"
  )
  "${TORCHRUN_CMD[@]}"
else
  TOTAL_WORKERS=$((NPROC_PER_NODE * WORKERS_PER_GPU))
  WORKER_LOG_DIR="${DATASET_ROOT}/worker_logs"
  mkdir -p "${WORKER_LOG_DIR}"
  declare -a PIDS=()
  for ((gpu_id=0; gpu_id<NPROC_PER_NODE; gpu_id++)); do
    for ((slot_id=0; slot_id<WORKERS_PER_GPU; slot_id++)); do
      global_rank=$((gpu_id * WORKERS_PER_GPU + slot_id))
      worker_log="${WORKER_LOG_DIR}/worker_${global_rank}.log"
      echo "Launching worker rank=${global_rank} on gpu=${gpu_id}, log=${worker_log}"
      CUDA_VISIBLE_DEVICES="${gpu_id}" \
      LOCAL_RANK=0 \
      RANK="${global_rank}" \
      WORLD_SIZE="${TOTAL_WORKERS}" \
      "${WAN_PYTHON}" "${RENDER_ARGS[@]}" > "${worker_log}" 2>&1 &
      PIDS+=("$!")
    done
  done

  fail=0
  for pid in "${PIDS[@]}"; do
    if ! wait "${pid}"; then
      fail=1
    fi
  done
  if [[ "${fail}" -ne 0 ]]; then
    echo "One or more workers failed."
    exit 1
  fi
fi

"${WAN_PYTHON}" "${ROOT_DIR}/scripts/build_metadata.py" \
  --dataset-dir "${DATASET_ROOT}" \
  --require-video

"${WAN_PYTHON}" "${ROOT_DIR}/scripts/verify_teacher_dataset.py" \
  --dataset-dir "${DATASET_ROOT}"
