#!/bin/bash
# Phase 6a step 6a.7 (D-E3, RQ4): the validation grid of fine-tuned Chronos-2 on the GPU node. One harness run per
# config, seed 0, 28-day context:
# - per sensor: the 8 configs of finetune_models.finetune_grid() on all 6 sensors (one fit per sensor);
# - pooled: the same 8 configs with --pooled on the 3 weather stations and the 2 halls (one fit per kind).
# 16 runs. The pick per sensor is made across both (select_on_validation.py, family "chronos2_ft").
# There is no scheduler and the 4 A100s are shared with other users: the job takes the GPU with the most free memory
# and refuses to start when it has less than MIN_FREE_MIB (default 30000; full fine-tuning at batch 256 needs far more
# than the DL models). Checkpoints go to TMPDIR on /scratch and are deleted after each fit. It stops at the first run
# that exits with an error, failed to fit, or has a failed origin (the runner records those in run.json, exit 0).
# GATE=1 runs the gates only:
# - gate Z: chronos2_ft_lr0 (learning rate 0, 10 steps) on the 5 main sensors must forecast like the zero-shot Mac
#   run (scripts/compare_runs.py, 28-day rows, the gate E rule);
# - the first grid run twice: its run time projects the grid (gate T), its two forecasts_sha256 show whether the GPU
#   fit repeats with the same seed (gate D), and run.json records the peak GPU memory per fit.
# The gate runs are evidence only: the grid runs that config again, and only the grid's runs go into the selection.
# GPU-hours: not measured yet; the GATE=1 run gives the projection.
# Usage, on the GPU node from the deployed project folder; start it under nohup so it survives a dropped ssh session:
#   GATE=1 nohup bash hpc_setup/jobs/finetune_validation.sh /scratch/ruha6285/benchmark_data > <log> 2>&1 &
set -eo pipefail
shopt -s inherit_errexit  # a failed run inside $(...) also stops the job
MAC_RUN_NAME=2026-10-14T222254Z_chronos2  # zero-shot Chronos-2 validation run from the Mac (in mac_runs/, gate E)
if [ $# -ne 1 ] || [ ! -f "$1/snapshots/sensbee-2026-10-01/manifest.json" ]; then
    echo "USAGE (on the GPU node): bash hpc_setup/jobs/finetune_validation.sh <benchmark_data folder>"
    echo "  the benchmark_data folder must hold snapshots/sensbee-2026-10-01"
    exit 1
fi
DATA=$(cd "$1" && pwd)  # absolute, as the harness runs from the project folder
cd "$(dirname "$0")/../.."
MIN_FREE_MIB=${MIN_FREE_MIB:-30000}
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "NO nvidia-smi: run this on the GPU node (makalu86)"; exit 1
fi
read -r GPU FREE < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | tr -d ' ' \
    | sort -t, -k2 -nr | head -1 | tr ',' ' ')
if [ "$FREE" -lt "$MIN_FREE_MIB" ]; then
    echo "NO GPU WITH $MIN_FREE_MIB MiB FREE (best: GPU $GPU with $FREE MiB); try again later"; exit 1
fi
export CUDA_DEVICE_ORDER=PCI_BUS_ID  # CUDA numbers the GPUs as nvidia-smi does
export CUDA_VISIBLE_DEVICES=$GPU
export HF_HUB_OFFLINE=1
# The pinned weights are in the deployed folder's cache (download_models.py --pinned); a fresh ssh shell has no HF_HOME.
export HF_HOME=${HF_HOME:-/scratch/ruha6285/sensbee_nvp/hf_cache}
export TMPDIR=/scratch/ruha6285/tmp  # fine-tuning checkpoints (~478 MB each, deleted after the fit)
mkdir -p "$TMPDIR"
echo "$(date -u +%FT%TZ) GPU $GPU ($FREE MiB free), commit $(cat COMMIT)"
# Without CUDA, the fine-tuning would quietly run on the CPU.
venv-bench/bin/python -c "import sys, torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); sys.exit(not torch.cuda.is_available())" \
    || { echo "TORCH SEES NO GPU: stopping"; exit 1; }
venv-bench/bin/python -c "
import sys
from huggingface_hub import try_to_load_from_cache
from benchmark.models import CHRONOS2
found = try_to_load_from_cache(CHRONOS2[0], 'model.safetensors', revision=CHRONOS2[1])
print('WEIGHTS', found)
sys.exit(not isinstance(found, str))" || { echo "CHRONOS-2 WEIGHTS NOT IN $HF_HOME: stopping"; exit 1; }

# One harness run; prints a RUN line with the run time and the run folder, and stops the job when the run has a fit
# error or a failed origin. The harness output goes to the job's log through a copy of its stderr descriptor (on Linux,
# tee /dev/stderr would reopen the log file and overwrite it).
run_one() {
    local model=$1
    shift
    local started=$SECONDS out
    out=$(venv-bench/bin/python scripts/run_harness.py --snapshot "$DATA/snapshots/sensbee-2026-10-01" \
          --out "$DATA/results" --model "$model" --period validation --contexts 28 --seed 0 "$@" 2>&1 \
          | tee >(cat >&2) | grep "^RUN WRITTEN to " | tail -1)
    echo "RUN $model $* $((SECONDS - started))s ${out#RUN WRITTEN to }"
    venv-bench/bin/python - "${out#RUN WRITTEN to }" <<'EOF'
import json, sys
counts = json.load(open(sys.argv[1] + "/run.json"))["counts"]
failed = {s: sum(c["failed"] for c in n["contexts"].values()) for s, n in counts.items()}
errors = {s: n["fit"]["error"] for s, n in counts.items() if "error" in n.get("fit", {})}
if any(failed.values()) or errors:
    sys.exit(f"RUN HAS FAILURES: failed origins {failed}, fit errors {errors}")
print("FITS", {s: {k: n["fit"].get(k) for k in ("train_s", "peak_gpu_mib", "gpu", "bf16")} for s, n in counts.items()})
EOF
}

POOLED_SENSORS="MANEBACH_WEATHER_STATION STUTZERBACH_WEATHER_STATION FRAUENWALD_WEATHER_STATION EISHALLE SCHWIMMHALLE"
CONFIGS=$(venv-bench/bin/python -c "from benchmark.models import FINETUNED; print(*[n for n in FINETUNED if '_tuned_' in n])")
JOBS=()
for config in $CONFIGS; do
    JOBS+=("$config|" "$config|--pooled --sensors $POOLED_SENSORS")
done
echo "${#JOBS[@]} RUNS PLANNED"

# One planned run from its "config|extra arguments" entry (no extra arguments: all 6 sensors, one fit each).
run_job() {
    local config extra
    IFS='|' read -r config extra <<< "$1"
    # shellcheck disable=SC2086
    run_one "$config" $extra
}

if [ "${GATE:-0}" = 1 ]; then
    if [ ! -f "$DATA/mac_runs/$MAC_RUN_NAME/run.json" ]; then
        echo "GATE Z NEEDS THE MAC RUN $DATA/mac_runs/$MAC_RUN_NAME (copied for gate E)"; exit 1
    fi
    zero=$(run_one chronos2_ft_lr0 --sensors $POOLED_SENSORS); echo "$zero"
    venv-bench/bin/python scripts/compare_runs.py "$DATA/mac_runs/$MAC_RUN_NAME" "$(echo "$zero" | head -1 | awk '{print $NF}')" \
        --contexts 28d && echo "GATE Z PASSED" || { echo "GATE Z FAILED"; exit 1; }
    first=$(run_job "${JOBS[0]}"); echo "$first"
    second=$(run_job "${JOBS[0]}"); echo "$second"
    for line in "$first" "$second"; do
        venv-bench/bin/python -c "import json, sys; print('forecasts_sha256', json.load(open(sys.argv[1] + '/run.json'))['forecasts_sha256'])" \
            "$(echo "$line" | head -1 | awk '{print $NF}')"
    done
    # The first run trains 6 fits x 250 steps; the grid 6 x 5,000 (per sensor) + 2 x 5,000 (pooled) = 40,000 steps.
    echo "GATE DONE: when training dominates, the grid takes about 27 x the first run's time (40,000 vs 1,500 steps)"
    exit 0
fi
for job in "${JOBS[@]}"; do
    run_job "$job"
done
echo "$(date -u +%FT%TZ) ALL ${#JOBS[@]} RUNS DONE"
