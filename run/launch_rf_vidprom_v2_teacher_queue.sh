#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/vepfs-mlp2/c20250518/241506050/code/video_code/data_construct"

WAN_PYTHON="${WAN_PYTHON:-/vepfs-mlp2/c20250518/241506050/miniconda3/envs/wan/bin/python}"
MODEL_ROOT="${MODEL_ROOT:-/vepfs-mlp2/c20250518/241506050/ckpts/wan21/Wan2.1-T2V-1.3B}"
PROMPT_BANK_PATH="${PROMPT_BANK_PATH:-${ROOT_DIR}/artifacts/prompt_banks/rf_vidprom_prompt_bank_v2/train.jsonl}"
DATASET_ROOT="${DATASET_ROOT:-${ROOT_DIR}/artifacts/datasets/rf_vidprom_v2_wan13b_teacher_train}"

NPROC_PER_NODE="${NPROC_PER_NODE:-auto}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_ITEMS="${MAX_ITEMS:-}"
QUEUE_MAX_JOBS_PER_WORKER="${QUEUE_MAX_JOBS_PER_WORKER:-0}"
QUEUE_LEASE_SECONDS="${QUEUE_LEASE_SECONDS:-1800}"
QUEUE_HEARTBEAT_SECONDS="${QUEUE_HEARTBEAT_SECONDS:-30}"
QUEUE_IDLE_SLEEP="${QUEUE_IDLE_SLEEP:-10}"
QUEUE_CLEANUP_STALE_LOCKS="${QUEUE_CLEANUP_STALE_LOCKS:-1}"
RUN_FINALIZE="${RUN_FINALIZE:-1}"
DRY_RUN="${DRY_RUN:-0}"

HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-832}"
NUM_FRAMES="${NUM_FRAMES:-81}"
FPS="${FPS:-16}"
QUALITY="${QUALITY:-5}"
OUTPUT_EXT="${OUTPUT_EXT:-mp4}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
CFG_SCALE="${CFG_SCALE:-6.0}"
SIGMA_SHIFT="${SIGMA_SHIFT:-8.0}"
SEED_OFFSET="${SEED_OFFSET:-0}"

if [[ ! -x "${WAN_PYTHON}" ]]; then
  echo "WAN_PYTHON is not executable: ${WAN_PYTHON}" >&2
  exit 2
fi

if [[ "${NPROC_PER_NODE}" == "auto" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    NPROC_PER_NODE="$(nvidia-smi -L | wc -l)"
  else
    NPROC_PER_NODE="1"
  fi
fi

if [[ "${NPROC_PER_NODE}" -lt 1 ]]; then
  NPROC_PER_NODE="1"
fi

mkdir -p "${DATASET_ROOT}/worker_logs"
cd "${ROOT_DIR}"

echo "Starting RF VidProM v2 queued teacher generation"
echo "ROOT_DIR=${ROOT_DIR}"
echo "WAN_PYTHON=${WAN_PYTHON}"
echo "MODEL_ROOT=${MODEL_ROOT}"
echo "PROMPT_BANK_PATH=${PROMPT_BANK_PATH}"
echo "DATASET_ROOT=${DATASET_ROOT}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "WORKERS_PER_GPU=${WORKERS_PER_GPU}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "QUEUE_LEASE_SECONDS=${QUEUE_LEASE_SECONDS}"
echo "RUN_FINALIZE=${RUN_FINALIZE}"
echo "DRY_RUN=${DRY_RUN}"

RENDER_ARGS=(
  "${ROOT_DIR}/scripts/render_teacher_dataset_queue.py"
  --prompt-bank "${PROMPT_BANK_PATH}"
  --output-dir "${DATASET_ROOT}"
  --model-root "${MODEL_ROOT}"
  --height "${HEIGHT}"
  --width "${WIDTH}"
  --num-frames "${NUM_FRAMES}"
  --fps "${FPS}"
  --quality "${QUALITY}"
  --output-ext "${OUTPUT_EXT}"
  --batch-size "${BATCH_SIZE}"
  --num-inference-steps "${NUM_INFERENCE_STEPS}"
  --cfg-scale "${CFG_SCALE}"
  --sigma-shift "${SIGMA_SHIFT}"
  --seed-offset "${SEED_OFFSET}"
  --lease-seconds "${QUEUE_LEASE_SECONDS}"
  --heartbeat-seconds "${QUEUE_HEARTBEAT_SECONDS}"
  --idle-sleep "${QUEUE_IDLE_SLEEP}"
)

if [[ -n "${MAX_ITEMS}" ]]; then
  RENDER_ARGS+=(--max-items "${MAX_ITEMS}")
fi

if [[ "${QUEUE_MAX_JOBS_PER_WORKER}" != "0" ]]; then
  RENDER_ARGS+=(--max-jobs "${QUEUE_MAX_JOBS_PER_WORKER}")
fi

if [[ "${QUEUE_CLEANUP_STALE_LOCKS}" == "1" ]]; then
  RENDER_ARGS+=(--cleanup-stale-locks)
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  RENDER_ARGS+=(--dry-run)
fi

declare -a PIDS=()
TOTAL_WORKERS=$((NPROC_PER_NODE * WORKERS_PER_GPU))
HOST_ID="$(hostname 2>/dev/null || echo unknown_host)"
TASK_ID="${VOLCANO_TASK_ID:-${VC_TASK_INDEX:-${CUSTOM_TASK_ID:-manual}}}"

for ((gpu_id=0; gpu_id<NPROC_PER_NODE; gpu_id++)); do
  for ((slot_id=0; slot_id<WORKERS_PER_GPU; slot_id++)); do
    global_slot=$((gpu_id * WORKERS_PER_GPU + slot_id))
    worker_id="${TASK_ID}_${HOST_ID}_gpu${gpu_id}_slot${slot_id}_pid$$_${global_slot}"
    worker_log="${DATASET_ROOT}/worker_logs/${worker_id}.log"
    echo "Launching queue worker ${worker_id} on CUDA_VISIBLE_DEVICES=${gpu_id}; log=${worker_log}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" \
    LOCAL_RANK=0 \
    RANK="${global_slot}" \
    WORLD_SIZE="${TOTAL_WORKERS}" \
    QUEUE_WORKER_SLOT="${global_slot}" \
    "${WAN_PYTHON}" "${RENDER_ARGS[@]}" --worker-id "${worker_id}" > "${worker_log}" 2>&1 &
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
  echo "One or more queue workers failed." >&2
  exit 1
fi

STATUS_PATH="${DATASET_ROOT}/queue_status.json"
STATUS_ARGS=(
  "${ROOT_DIR}/scripts/render_teacher_dataset_queue.py"
  --prompt-bank "${PROMPT_BANK_PATH}"
  --output-dir "${DATASET_ROOT}"
  --output-ext "${OUTPUT_EXT}"
  --lease-seconds "${QUEUE_LEASE_SECONDS}"
  --status-only
)

if [[ -n "${MAX_ITEMS}" ]]; then
  STATUS_ARGS+=(--max-items "${MAX_ITEMS}")
fi

"${WAN_PYTHON}" "${STATUS_ARGS[@]}" > "${STATUS_PATH}"

echo "Queue status:"
cat "${STATUS_PATH}"

if [[ "${RUN_FINALIZE}" == "1" ]]; then
  "${WAN_PYTHON}" "${ROOT_DIR}/scripts/build_metadata.py" \
    --dataset-dir "${DATASET_ROOT}" \
    --require-video

  "${WAN_PYTHON}" "${ROOT_DIR}/scripts/verify_teacher_dataset.py" \
    --dataset-dir "${DATASET_ROOT}"
fi
