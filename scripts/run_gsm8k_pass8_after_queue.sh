#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
baseline=/root/autodl-tmp/ReasoningForge/runs/qwen15b_gsm8k_full_4090_fast_v2
log_dir=/root/autodl-tmp/logs
status="$log_dir/gsm8k_pass8_queue.status.log"
mkdir -p "$log_dir" "$project_dir/runs"
export HF_HOME=/root/autodl-tmp/hf-cache
export HF_HUB_OFFLINE=1
cd "$project_dir"

report() { printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$status"; }
trap 'report "FAILED at line $LINENO (exit $?)"' ERR

report "Waiting for existing experiment queue to release the GPU"
while tmux has-session -t dapo_experiment_queue 2>/dev/null; do sleep 60; done
while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]]; do
    report "GPU still occupied; retrying in 60 seconds"
    sleep 60
done

test -f "$baseline/checkpoint-001200/COMPLETE"
test -s "$baseline/checkpoint-001200/model/adapter_model.safetensors"
test -f runs/qwen15b_gsm8k_dynamic_sampling_full/checkpoint-001200/COMPLETE
test -s runs/qwen15b_gsm8k_dynamic_sampling_full/final/adapter_model.safetensors

run_job() {
    local label="$1"
    shift
    report "START $label"
    "$@" > "$log_dir/$label.log" 2>&1
    report "DONE $label"
}

run_job gsm8k_baseline_1200_dev500_8 .conda/bin/python -u evaluate.py \
    --config configs/qwen15b_gsm8k_full_4090_fast_v2.json \
    --model "$baseline/checkpoint-001200/model" --data data/gsm8k_dev.jsonl \
    --num-samples 8 --k 1 4 8 \
    --output runs/gsm8k_baseline_1200_dev500_8.json

run_job gsm8k_dynamic_dev500_8 .conda/bin/python -u evaluate.py \
    --config configs/qwen15b_gsm8k_dynamic_sampling_full.json \
    --model runs/qwen15b_gsm8k_dynamic_sampling_full/final \
    --data data/gsm8k_dev.jsonl --num-samples 8 --k 1 4 8 \
    --output runs/gsm8k_dynamic_dev500_8.json

run_job gsm8k_dynamic_1200_comparison_8 .conda/bin/python scripts/compare_ablation.py \
    --baseline-run "$baseline" --baseline-step 1200 \
    --variant-run runs/qwen15b_gsm8k_dynamic_sampling_full \
    --baseline-eval runs/gsm8k_baseline_1200_dev500_8.json \
    --variant-eval runs/gsm8k_dynamic_dev500_8.json \
    --output runs/gsm8k_dynamic_1200_comparison_8.json

report "PASS@1, PASS@4 AND PASS@8 COMPARISON COMPLETE"
