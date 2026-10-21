#!/usr/bin/env bash
set -euo pipefail

# Continue the currently running Dynamic Sampling job, then evaluate both
# training runs with the original four-sample protocol and with eight samples.
project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
baseline=/root/autodl-tmp/ReasoningForge/runs/qwen15b_gsm8k_full_4090_fast_v2
dynamic="$project_dir/runs/qwen15b_gsm8k_dynamic_sampling_full"
log_dir=/root/autodl-tmp/logs
status="$log_dir/gsm8k_4_and_8_queue.status.log"
: "${TRAIN_PID:?Set TRAIN_PID to the current Dynamic Sampling training process}"
: "${PAUSED_QUEUE_PID:?Set PAUSED_QUEUE_PID to the paused original queue shell}"
train_pid=$TRAIN_PID
paused_queue_pid=$PAUSED_QUEUE_PID
if [[ ! "$train_pid" =~ ^[0-9]+$ || ! "$paused_queue_pid" =~ ^[0-9]+$ ||
      "$train_pid" -le 1 || "$paused_queue_pid" -le 1 ]]; then
    printf 'TRAIN_PID and PAUSED_QUEUE_PID must be process IDs greater than 1\n' >&2
    exit 1
fi
train_command=$(ps -p "$train_pid" -o args= || true)
queue_command=$(ps -p "$paused_queue_pid" -o args= || true)
queue_state=$(ps -p "$paused_queue_pid" -o stat= || true)
train_parent=$(ps -p "$train_pid" -o ppid= | tr -d '[:space:]' || true)
if [[ "$train_command" != *"train.py --config configs/qwen15b_gsm8k_dynamic_sampling_full.json"* ||
      "$queue_command" != *"bash $project_dir/scripts/run_queued_dapo_experiments.sh"* ||
      "$queue_state" != T* || "$train_parent" != "$paused_queue_pid" ]]; then
    printf 'Process identities or paused queue state do not match the expected training handoff\n' >&2
    exit 1
fi
queue_started=$(ps -p "$paused_queue_pid" -o lstart=)

mkdir -p "$log_dir" "$project_dir/runs"
export HF_HOME=/root/autodl-tmp/hf-cache
export HF_HUB_OFFLINE=1
cd "$project_dir"

report() { printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$status"; }
trap 'report "FAILED at line $LINENO (exit $?)"' ERR

report "Waiting for Dynamic Sampling training to finish"
while true; do
    state=$(ps -p "$train_pid" -o stat= 2>/dev/null | tr -d '[:space:]' || true)
    if [[ -z "$state" || "$state" == Z* ]]; then
        break
    fi
    sleep 10
done

test -f "$baseline/checkpoint-001200/COMPLETE"
test -s "$baseline/checkpoint-001200/model/adapter_model.safetensors"
test -f "$dynamic/checkpoint-001200/COMPLETE"
test -s "$dynamic/final/adapter_model.safetensors"
test -s "$dynamic/final/tokenizer_config.json"

# The original queue is paused at its training child. The child has exited,
# so terminating only the paused shell skips duplicate evaluation and DAPO-Math.
if [[ "$(ps -p "$paused_queue_pid" -o lstart= || true)" == "$queue_started" ]]; then
    kill -KILL "$paused_queue_pid"
fi
report "Dynamic Sampling finished; original queue closed before evaluation and DAPO-Math"

while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]]; do
    report "GPU occupied; waiting 10 seconds"
    sleep 10
done

run_job() {
    local label="$1"
    shift
    report "START $label"
    "$@" > "$log_dir/$label.log" 2>&1
    report "DONE $label"
}

run_job gsm8k_baseline_1200_dev500 .conda/bin/python -u evaluate.py \
    --config configs/qwen15b_gsm8k_full_4090_fast_v2.json \
    --model "$baseline/checkpoint-001200/model" --data data/gsm8k_dev.jsonl \
    --num-samples 4 --k 1 4 --output runs/gsm8k_baseline_1200_dev500_4.json

run_job gsm8k_dynamic_dev500 .conda/bin/python -u evaluate.py \
    --config configs/qwen15b_gsm8k_dynamic_sampling_full.json \
    --model "$dynamic/final" --data data/gsm8k_dev.jsonl \
    --num-samples 4 --k 1 4 --output runs/gsm8k_dynamic_dev500_4.json

run_job gsm8k_dynamic_1200_comparison .conda/bin/python scripts/compare_ablation.py \
    --baseline-run "$baseline" --baseline-step 1200 \
    --variant-run "$dynamic" \
    --baseline-eval runs/gsm8k_baseline_1200_dev500_4.json \
    --variant-eval runs/gsm8k_dynamic_dev500_4.json \
    --output runs/gsm8k_dynamic_1200_comparison.json

run_job gsm8k_baseline_1200_dev500_8 .conda/bin/python -u evaluate.py \
    --config configs/qwen15b_gsm8k_full_4090_fast_v2.json \
    --model "$baseline/checkpoint-001200/model" --data data/gsm8k_dev.jsonl \
    --num-samples 8 --k 1 4 8 --output runs/gsm8k_baseline_1200_dev500_8.json

run_job gsm8k_dynamic_dev500_8 .conda/bin/python -u evaluate.py \
    --config configs/qwen15b_gsm8k_dynamic_sampling_full.json \
    --model "$dynamic/final" --data data/gsm8k_dev.jsonl \
    --num-samples 8 --k 1 4 8 --output runs/gsm8k_dynamic_dev500_8.json

run_job gsm8k_dynamic_1200_comparison_8 .conda/bin/python scripts/compare_ablation.py \
    --baseline-run "$baseline" --baseline-step 1200 \
    --variant-run "$dynamic" \
    --baseline-eval runs/gsm8k_baseline_1200_dev500_8.json \
    --variant-eval runs/gsm8k_dynamic_dev500_8.json \
    --output runs/gsm8k_dynamic_1200_comparison_8.json

if [[ "${RUN_DAPO_MATH_PILOT:-0}" != 1 ]]; then
    report "GSM8K comparisons complete; DAPO-Math pilot disabled"
    exit 0
fi
report "GSM8K 4-sample and 8-sample comparisons complete; starting DAPO-Math baseline pilot"

run_job dapo_math_baseline_pilot .conda/bin/python -u train.py \
    --config configs/qwen15b_dapo_math_17k_baseline_pilot.json
test -f runs/qwen15b_dapo_math_17k_baseline_pilot/checkpoint-000200/COMPLETE
test -s runs/qwen15b_dapo_math_17k_baseline_pilot/final/adapter_model.safetensors

mkdir -p exports
archive=exports/dapo_math_baseline_pilot.tar.gz
run_job dapo_math_baseline_export tar -czf "$archive.partial" \
    runs/qwen15b_dapo_math_17k_baseline_pilot \
    configs/qwen15b_dapo_math_17k_baseline_pilot.json \
    data/dapo_math_17k_train.jsonl \
    data/dapo_math_17k_dev.jsonl \
    data/dapo_math_17k_test.jsonl \
    data/dapo_math_17k_manifest.json
mv "$archive.partial" "$archive"
sha256sum "$archive" > "$archive.sha256"
report "DAPO-Math pilot model and dataset export ready: $archive"

run_job dapo_math_baseline_dev500 .conda/bin/python -u evaluate.py \
    --config configs/qwen15b_dapo_math_17k_baseline_pilot.json \
    --model runs/qwen15b_dapo_math_17k_baseline_pilot/final \
    --data data/dapo_math_17k_dev.jsonl --num-samples 4 --k 1 4 \
    --output runs/dapo_math_baseline_dev500_4.json

report "GSM8K comparisons and DAPO-Math baseline pilot complete"
