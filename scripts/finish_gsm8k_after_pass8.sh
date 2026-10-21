#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
baseline=/root/autodl-tmp/ReasoningForge/runs/qwen15b_gsm8k_full_4090_fast_v2
dynamic="$project_dir/runs/qwen15b_gsm8k_dynamic_sampling_full"
log_dir=/root/autodl-tmp/logs
status="$log_dir/gsm8k_finish_after_pass8.status.log"
: "${EVAL_PID:?Set EVAL_PID to the current Dynamic Sampling pass@8 evaluation process}"
: "${PAUSED_QUEUE_PID:?Set PAUSED_QUEUE_PID to the paused GSM8K queue shell}"
eval_pid=$EVAL_PID
paused_queue_pid=$PAUSED_QUEUE_PID
if [[ ! "$eval_pid" =~ ^[0-9]+$ || ! "$paused_queue_pid" =~ ^[0-9]+$ ||
      "$eval_pid" -le 1 || "$paused_queue_pid" -le 1 ]]; then
    printf 'EVAL_PID and PAUSED_QUEUE_PID must be process IDs greater than 1\n' >&2
    exit 1
fi
eval_command=$(ps -p "$eval_pid" -o args= || true)
queue_command=$(ps -p "$paused_queue_pid" -o args= || true)
queue_state=$(ps -p "$paused_queue_pid" -o stat= || true)
eval_parent=$(ps -p "$eval_pid" -o ppid= | tr -d '[:space:]' || true)
if [[ "$eval_command" != *"evaluate.py --config configs/qwen15b_gsm8k_dynamic_sampling_full.json"* ||
      "$eval_command" != *"--output runs/gsm8k_dynamic_dev500_8.json"* ||
      "$queue_command" != *"bash $project_dir/scripts/run_gsm8k_4_and_8_after_dynamic.sh"* ||
      "$queue_state" != T* || "$eval_parent" != "$paused_queue_pid" ]]; then
    printf 'Process identities or paused queue state do not match the expected evaluation handoff\n' >&2
    exit 1
fi
queue_started=$(ps -p "$paused_queue_pid" -o lstart=)
mkdir -p "$log_dir"
cd "$project_dir"

report() { printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$status"; }
close_old_queue() {
    if [[ "$(ps -p "$paused_queue_pid" -o lstart= || true)" == "$queue_started" ]]; then
        kill -KILL "$paused_queue_pid"
    fi
}
shutdown_after_sync() {
    if [[ "${AUTO_SHUTDOWN:-0}" != 1 ]]; then
        report "Automatic shutdown disabled; set AUTO_SHUTDOWN=1 explicitly to enable it"
        return
    fi
    for ((attempt=1; attempt<=30; attempt++)); do
        if [[ -f runs/GSM8K_LOCAL_SYNC_COMPLETE ]]; then
            report "Local results confirmed; requesting AutoDL shutdown"
            /usr/bin/shutdown
            return
        fi
        sleep 10
    done
    report "Local sync unconfirmed after five minutes; results remain on remote disk; requesting AutoDL shutdown to stop billing"
    /usr/bin/shutdown
}

report "Waiting for current Dynamic Sampling pass@8 evaluation; DAPO-Math canceled"
while true; do
    state=$(ps -p "$eval_pid" -o stat= 2>/dev/null | tr -d '[:space:]' || true)
    if [[ -z "$state" || "$state" == Z* ]]; then
        break
    fi
    sleep 10
done

if [[ ! -s runs/gsm8k_dynamic_dev500_8.json ]]; then
    report "FAILED: pass@8 evaluation did not produce a result"
    close_old_queue
    shutdown_after_sync
    exit 1
fi

report "Dynamic Sampling pass@8 evaluation finished; comparing both models"
if ! .conda/bin/python scripts/compare_ablation.py \
    --baseline-run "$baseline" --baseline-step 1200 \
    --variant-run "$dynamic" \
    --baseline-eval runs/gsm8k_baseline_1200_dev500_8.json \
    --variant-eval runs/gsm8k_dynamic_dev500_8.json \
    --output runs/gsm8k_dynamic_1200_comparison_8.json \
    > "$log_dir/gsm8k_dynamic_1200_comparison_8.log" 2>&1; then
    report "FAILED: pass@8 comparison could not be generated"
    close_old_queue
    shutdown_after_sync
    exit 1
fi

test -s runs/gsm8k_dynamic_1200_comparison_8.json
sha256sum \
    runs/gsm8k_baseline_1200_dev500_4.json \
    runs/gsm8k_dynamic_dev500_4.json \
    runs/gsm8k_dynamic_1200_comparison.json \
    runs/gsm8k_baseline_1200_dev500_8.json \
    runs/gsm8k_dynamic_dev500_8.json \
    runs/gsm8k_dynamic_1200_comparison_8.json \
    > runs/gsm8k_final_results.sha256
report "GSM8K result files and checksums saved; stopping experiment queue"
close_old_queue
report "ALL GSM8K JOBS COMPLETE; NO DAPO-MATH TRAINING STARTED"
shutdown_after_sync
