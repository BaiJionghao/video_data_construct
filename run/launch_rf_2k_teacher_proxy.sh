#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/vepfs-mlp2/c20250518/241506050/code/video_code/data_construct"

export DATASET_ROOT="${DATASET_ROOT:-${ROOT_DIR}/artifacts/datasets/rf_wan13b_2k_teacher_proxy}"
export PROMPT_BANK_DIR="${PROMPT_BANK_DIR:-${ROOT_DIR}/artifacts/prompt_banks/rf_prompt_bank_v1}"
export MODEL_ROOT="${MODEL_ROOT:-/vepfs-mlp2/c20250518/241506050/ckpts/wan21/Wan2.1-T2V-1.3B}"
export WAN_PYTHON="${WAN_PYTHON:-/vepfs-mlp2/c20250518/241506050/miniconda3/envs/wan/bin/python}"

export TRAIN_COUNT="${TRAIN_COUNT:-20000}"
export VAL_COUNT="${VAL_COUNT:-1000}"
export PROMPT_SPLIT="${PROMPT_SPLIT:-train}"
export MAX_ITEMS="${MAX_ITEMS:-2000}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export WORKERS_PER_GPU="${WORKERS_PER_GPU:-1}"
export BATCH_SIZE="${BATCH_SIZE:-1}"

cd "${ROOT_DIR}"

echo "Starting RF 2k teacher proxy generation"
echo "DATASET_ROOT=${DATASET_ROOT}"
echo "PROMPT_BANK_DIR=${PROMPT_BANK_DIR}"
echo "MODEL_ROOT=${MODEL_ROOT}"
echo "MAX_ITEMS=${MAX_ITEMS}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "WORKERS_PER_GPU=${WORKERS_PER_GPU}"
echo "BATCH_SIZE=${BATCH_SIZE}"

bash "${ROOT_DIR}/run/launch_teacher_dataset_8gpu.sh"
