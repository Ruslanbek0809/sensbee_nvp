#!/bin/bash
# Phase 6a step 6a.3 (D-E5): the validation grid of the tuned DeepAR or PatchTST configs on the GPU node. One harness
# run per config, seed 0, 28-day context only (the pick per sensor is made at 28 days; the picks then run on test).
# - deepar: 4 configs, each without calendar inputs (6 sensors) and with them: cal on the halls, sun on the
#   stations, cal_parking on parking. 16 runs.
# - patchtst: 12 configs, no covariates (6 sensors). 12 runs.
# There is no scheduler and the 4 A100s are shared with other users: the job takes the GPU with the most free memory
# and refuses to start when it has less than MIN_FREE_MIB (default 8000). It stops at the first run that exits with an
# error, failed to fit, or has a failed origin (the runner records those in run.json and still exits 0).
# GATE=1 runs only the first run, twice: its run time projects the grid (gate T) and the two forecasts_sha256 show
# whether the GPU fit gives the same forecasts with the same seed (gate D). The gate runs are evidence only: the full
# grid runs that config again, and only the grid's runs go into the selection (select_on_validation.py refuses two runs
# of one model).
# GPU-hours: not measured yet; the GATE=1 run gives the projection. On the Mac CPU the 100-epoch defaults took
# 37 min (DeepAR) and 23 min (PatchTST) for 6 sensors and 2 contexts; the tuned configs allow up to 300 epochs.
# Usage, on the GPU node from the deployed project folder; start it under nohup so it survives a dropped ssh session:
#   GATE=1 nohup bash hpc_setup/jobs/tuned_dl_validation.sh /scratch/ruha6285/benchmark_data deepar > <log> 2>&1 &
set -eo pipefail
shopt -s inherit_errexit  # a failed run inside $(...) also stops the job
if [ $# -ne 2 ] || [ ! -f "$1/snapshots/sensbee-2026-10-01/manifest.json" ] || { [ "$2" != deepar ] && [ "$2" != patchtst ]; }; then
    echo "USAGE (on the GPU node): bash hpc_setup/jobs/tuned_dl_validation.sh <benchmark_data folder> <deepar|patchtst>"
    echo "  the benchmark_data folder must hold snapshots/sensbee-2026-10-01"
    exit 1
fi
DATA=$(cd "$1" && pwd)  # absolute, as the harness runs from the project folder
cd "$(dirname "$0")/../.."
FAMILY=$2
MIN_FREE_MIB=${MIN_FREE_MIB:-8000}
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
echo "$(date -u +%FT%TZ) GPU $GPU ($FREE MiB free), commit $(cat COMMIT)"
# Without CUDA, AutoGluon would quietly train on the CPU.
venv-bench/bin/python -c "import sys, torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); sys.exit(not torch.cuda.is_available())" \
    || { echo "TORCH SEES NO GPU: stopping"; exit 1; }

# One harness run; prints a RUN line with the run time and the run folder, and stops the job when the run has a fit
# error or a failed origin. The harness output goes to the job's log through a copy of its stderr descriptor (on Linux,
# tee /dev/stderr would reopen the log file and overwrite it).
run_one() {
    local model=$1 set=$2
    shift 2
    local args=(--snapshot "$DATA/snapshots/sensbee-2026-10-01" --out "$DATA/results" --model "$model"
                --period validation --contexts 28 --seed 0)
    [ -n "$set" ] && args+=(--covariates "$set")
    local started=$SECONDS out
    out=$(venv-bench/bin/python scripts/run_harness.py "${args[@]}" "$@" 2>&1 | tee >(cat >&2) \
          | grep "^RUN WRITTEN to " | tail -1)
    echo "RUN $model ${set:-values} $((SECONDS - started))s ${out#RUN WRITTEN to }"
    venv-bench/bin/python - "${out#RUN WRITTEN to }" <<'EOF'
import json, sys
counts = json.load(open(sys.argv[1] + "/run.json"))["counts"]
failed = {s: sum(c["failed"] for c in n["contexts"].values()) for s, n in counts.items()}
errors = {s: n["fit"]["error"] for s, n in counts.items() if "error" in n.get("fit", {})}
if any(failed.values()) or errors:
    sys.exit(f"RUN HAS FAILURES: failed origins {failed}, fit errors {errors}")
EOF
}

CONFIGS=$(venv-bench/bin/python -c "from benchmark.models import TUNED; print(*[n for n in TUNED if n.startswith('ag_${FAMILY}_tuned_')])")
JOBS=()
for config in $CONFIGS; do
    JOBS+=("$config|")
    if [ "$FAMILY" = deepar ]; then
        JOBS+=("$config|cal|EISHALLE SCHWIMMHALLE"
               "$config|sun|MANEBACH_WEATHER_STATION STUTZERBACH_WEATHER_STATION FRAUENWALD_WEATHER_STATION"
               "$config|cal_parking|PARKPLATZ_EAZ")
    fi
done
echo "${#JOBS[@]} RUNS PLANNED"

# One planned run from its "config|set|sensors" entry (no sensors: all 6).
run_job() {
    local config set sensors
    IFS='|' read -r config set sensors <<< "$1"
    if [ -n "$sensors" ]; then
        # shellcheck disable=SC2086
        run_one "$config" "$set" --sensors $sensors
    else
        run_one "$config" "$set"
    fi
}

if [ "${GATE:-0}" = 1 ]; then
    first=$(run_job "${JOBS[0]}"); echo "$first"
    second=$(run_job "${JOBS[0]}"); echo "$second"
    for line in "$first" "$second"; do
        venv-bench/bin/python -c "import json, sys; print('forecasts_sha256', json.load(open(sys.argv[1] + '/run.json'))['forecasts_sha256'])" \
            "$(echo "$line" | awk '{print $5}')"
    done
    echo "GATE DONE: projected grid time = ${#JOBS[@]} x the first run's time (runs differ in context and sensors)"
    exit 0
fi
for job in "${JOBS[@]}"; do
    run_job "$job"
done
echo "$(date -u +%FT%TZ) ALL ${#JOBS[@]} RUNS DONE"
