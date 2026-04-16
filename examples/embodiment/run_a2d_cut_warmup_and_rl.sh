#!/usr/bin/env bash

set -euo pipefail

WORKSPACE_ROOT="/home/psibot/workspace_zhiyuan"
RLINF_ROOT="${WORKSPACE_ROOT}/RLinf"
PSI_POLICY_ROOT="${WORKSPACE_ROOT}/psi-policy"
PYTHON_BIN="${WORKSPACE_ROOT}/miniconda/envs/a2d/bin/python"

INPUT_ZARR="${INPUT_ZARR:-${PSI_POLICY_ROOT}/data/bj01-dc09_20260409_22.zarr}"
CUT_OUTPUT_PREFIX="${CUT_OUTPUT_PREFIX:-${PSI_POLICY_ROOT}/data/bj01-dc09_20260409_22_cut}"
REPLAY_BUFFER_DIR="${REPLAY_BUFFER_DIR:-${WORKSPACE_ROOT}/offline_buffers/bj01-dc09_20260409_22_cut}"
WARMUP_DIR="${WARMUP_DIR:-${WORKSPACE_ROOT}/offline_warmup/bj01-dc09_20260409_22}"
RESULTS_DIR="${RESULTS_DIR:-${WORKSPACE_ROOT}/results/a2d_sac_psi_20260409_22}"
WARMUP_UPDATES="${WARMUP_UPDATES:-1000}"
DISPLAY_VALUE="${DISPLAY:-:1}"

export EMBODIED_PATH="${RLINF_ROOT}/examples/embodiment"
export PYTHONPATH="${RLINF_ROOT}"
export DISPLAY="${DISPLAY_VALUE}"

mkdir -p "$(dirname "${REPLAY_BUFFER_DIR}")" "$(dirname "${WARMUP_DIR}")" "$(dirname "${RESULTS_DIR}")"

echo "[1/4] Launching clip cutter on ${INPUT_ZARR}"
echo "      Output prefix: ${CUT_OUTPUT_PREFIX}"
echo "      Use S to mark start, E to save clip, O to save current episode, Q to close."

cd "${PSI_POLICY_ROOT}"
"${PYTHON_BIN}" tools/vis_and_cut_demo.py \
  --input "${INPUT_ZARR}" \
  --output "${CUT_OUTPUT_PREFIX}" \
  --keep-clips-as-episodes \
  --overwrite

shopt -s nullglob
cut_files=( "${CUT_OUTPUT_PREFIX}"_ep*.zarr )
shopt -u nullglob

if [ "${#cut_files[@]}" -eq 0 ]; then
  echo "No clipped zarr files were saved under ${CUT_OUTPUT_PREFIX}_ep*.zarr" >&2
  exit 1
fi

echo "[2/4] Converting ${#cut_files[@]} clipped zarr file(s) into RLinf replay buffer"
cd "${RLINF_ROOT}"
"${PYTHON_BIN}" examples/embodiment/convert_a2d_zarr_to_replay_buffer.py \
  --zarr-path "${cut_files[@]}" \
  --output-dir "${REPLAY_BUFFER_DIR}" \
  --model-weights-id "offline_success_demo_20260409_22" \
  --overwrite

echo "[3/4] Running offline critic warmup for ${WARMUP_UPDATES} update(s)"
"${PYTHON_BIN}" examples/embodiment/train_embodied_agent.py \
  --config-name realworld_a2d_sac_psi \
  algorithm.replay_buffer.load_path="${REPLAY_BUFFER_DIR}" \
  runner.offline_critic_warmup_updates="${WARMUP_UPDATES}" \
  runner.offline_critic_warmup_save_dir="${WARMUP_DIR}" \
  runner.logger.log_path="${RESULTS_DIR}"

echo "[4/4] Starting online real-world RL exploration from warmup checkpoint"
"${PYTHON_BIN}" examples/embodiment/train_embodied_agent.py \
  --config-name realworld_a2d_sac_psi \
  runner.resume_dir="${WARMUP_DIR}" \
  runner.logger.log_path="${RESULTS_DIR}"
