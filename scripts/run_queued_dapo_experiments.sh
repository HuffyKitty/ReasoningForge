#!/usr/bin/env bash
set -euo pipefail

# Run this only in a detached tmux session after stopping the GSM8K baseline
# at its saved step-1200 checkpoint. GPU jobs run one at a time.
project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
original_dir=/root/autodl-tmp/ReasoningForge
log_dir=/root/autodl-tmp/logs
status="$log_dir/dapo_experiment_queue.status.log"
mkdir -p "$log_dir" "$project_dir/runs"
export HF_HOME=/root/autodl-tmp/hf-cache
export HF_HUB_OFFLINE=1
cd "$project_dir"

report() { printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$status"; }
trap 'report "FAILED at line $LINENO (exit $?)"' ERR

baseline="$original_dir/runs/qwen15b_gsm8k_full_4090_fast_v2"
baseline_model="$baseline/checkpoint-001200/model"
if tmux has-session -t reasoningforge_full_v2 2>/dev/null; then
    report "Baseline training is still running; refusing to launch another GPU job"
    exit 1
fi
if [[ ! -f "$baseline/checkpoint-001200/COMPLETE" ||
      ! -s "$baseline_model/adapter_model.safetensors" ]]; then
    report "Baseline step-1200 checkpoint is incomplete; refusing to launch GPU jobs"
    exit 1
fi
report "Baseline stopped at selected checkpoint-001200; starting GSM8K Dynamic Sampling"

run_job() {
    local label="$1"
    local logfile="$log_dir/$label.log"
    shift
    report "START $label"
    "$@" > "$logfile" 2>&1
    report "DONE $label"
}

run_job gsm8k_dynamic_smoke .conda/bin/python -u train.py \
    --config configs/qwen15b_gsm8k_dynamic_sampling_smoke.json
test -f runs/qwen15b_gsm8k_dynamic_sampling_smoke/checkpoint-000002/COMPLETE

run_job gsm8k_dynamic_full .conda/bin/python -u train.py \
    --config configs/qwen15b_gsm8k_dynamic_sampling_full.json
test -f runs/qwen15b_gsm8k_dynamic_sampling_full/final/adapter_model.safetensors

run_job gsm8k_baseline_1200_dev500 .conda/bin/python -u evaluate.py \
    --config configs/qwen15b_gsm8k_full_4090_fast_v2.json \
    --model "$baseline_model" --data data/gsm8k_dev.jsonl \
    --num-samples 4 --k 1 4 --output runs/gsm8k_baseline_1200_dev500_4.json
run_job gsm8k_dynamic_dev500 .conda/bin/python -u evaluate.py \
    --config configs/qwen15b_gsm8k_dynamic_sampling_full.json \
    --model runs/qwen15b_gsm8k_dynamic_sampling_full/final \
    --data data/gsm8k_dev.jsonl --num-samples 4 --k 1 4 \
    --output runs/gsm8k_dynamic_dev500_4.json
run_job gsm8k_dynamic_1200_comparison .conda/bin/python scripts/compare_ablation.py \
    --baseline-run "$baseline" \
    --baseline-step 1200 \
    --variant-run runs/qwen15b_gsm8k_dynamic_sampling_full \
    --baseline-eval runs/gsm8k_baseline_1200_dev500_4.json \
    --variant-eval runs/gsm8k_dynamic_dev500_4.json \
    --output runs/gsm8k_dynamic_1200_comparison.json

if [[ "${RUN_DAPO_MATH_PILOT:-0}" != 1 ]]; then
    report "GSM8K comparison complete; DAPO-Math pilot disabled"
    exit 0
fi
run_job dapo_math_baseline_pilot .conda/bin/python -u train.py \
    --config configs/qwen15b_dapo_math_17k_baseline_pilot.json
test -f runs/qwen15b_dapo_math_17k_baseline_pilot/final/adapter_model.safetensors
run_job dapo_math_baseline_dev500 .conda/bin/python -u evaluate.py \
    --config configs/qwen15b_dapo_math_17k_baseline_pilot.json \
    --model runs/qwen15b_dapo_math_17k_baseline_pilot/final \
    --data data/dapo_math_17k_dev.jsonl --num-samples 4 --k 1 4 \
    --output runs/dapo_math_baseline_dev500_4.json
report "ALL REQUESTED QUEUED JOBS COMPLETE"
