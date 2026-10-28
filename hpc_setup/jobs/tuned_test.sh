#!/bin/bash
# Phase 6a steps 6a.3 and 6a.7: the test runs of the configs picked on validation (D-E5, D-E3), on the GPU node.
# The picks come from a selection.csv of scripts/select_on_validation.py (rows with selected = True), copied to the
# cluster. One harness run per picked config and seed, on the sensors that picked it, period test:
# - the run name in the "model" column gives the registry entry, the covariate set after "+" and --pooled for a name
#   ending in "_pooled";
# - a pooled pick runs on every sensor of its sensors' kinds (e.g. all 3 weather stations), so the test fit pools the
#   same sensors as the validation fit; only the picking sensors' rows are its result;
# - contexts: the registry's own (14 and 28 days when the config's input window fits 14 days, else 28);
# - seeds: SEEDS (default "0 1 2 3 4"); each sensor's result is the seed-averaged per-origin loss.
# There is no scheduler and the 4 A100s are shared with other users: the job takes the GPU with the most free memory
# and refuses to start when it has less than MIN_FREE_MIB (default 30000 when a fine-tuned Chronos-2 is picked, else
# 8000). It stops at the first run that exits with an error, failed to fit, or has a failed origin (the runner records
# those in run.json, exit 0). PLAN_ONLY=1 prints the planned runs and stops before any GPU check.
# Only DeepAR, PatchTST and fine-tuned Chronos-2 picks are accepted (tabular test runs: Mac, seed 0).
# GPU-hours, not measured: the validation runs on the shared node took 0.3-0.9 h (PatchTST), 0.1-2.3 h (DeepAR) and
# 0.9-3.6 h (fine-tuning, 6 sensors) each; 5 seeds of every pick cost about 5 x one validation run per pick, more for
# the extra 14-day forecasts.
# Usage, on the GPU node from the deployed project folder; start it under nohup so it survives a dropped ssh session:
#   nohup bash hpc_setup/jobs/tuned_test.sh /scratch/ruha6285/benchmark_data <selection.csv> > <log> 2>&1 &
set -eo pipefail
shopt -s inherit_errexit  # a failed run inside $(...) also stops the job
if [ $# -ne 2 ] || [ ! -f "$1/snapshots/sensbee-2026-10-01/manifest.json" ] || [ ! -f "$2" ]; then
    echo "USAGE (on the GPU node): bash hpc_setup/jobs/tuned_test.sh <benchmark_data folder> <selection.csv>"
    echo "  the benchmark_data folder must hold snapshots/sensbee-2026-10-01"
    exit 1
fi
DATA=$(cd "$1" && pwd)  # absolute, as the harness runs from the project folder
PICKS=$(cd "$(dirname "$2")" && pwd)/$(basename "$2")
cd "$(dirname "$0")/../.."
SEEDS=${SEEDS:-0 1 2 3 4}

# The planned runs, one "registry name|covariate set|pooled (0/1)|sensors" line per picked config.
PLANNED=$(venv-bench/bin/python - "$PICKS" <<'EOF'
import sys
import pandas as pd
from benchmark.models import FINETUNED, TUNED
from benchmark.tasks import TASKS
picks = pd.read_csv(sys.argv[1])
picks = picks[picks["selected"]]
if picks.empty:
    sys.exit("NO SELECTED ROWS IN THE PICKS FILE")
for name, rows in picks.groupby("model", sort=True):
    pooled = name.endswith("_pooled")
    model, _, covariates = name.removesuffix("_pooled").partition("+")
    # only the GPU families; the tuned tabular models ran their test runs on the Mac with seed 0
    if not (model.startswith(("ag_deepar_tuned_", "ag_patchtst_tuned_")) and model in TUNED
            or model.startswith("chronos2_ft_tuned_") and model in FINETUNED):
        sys.exit(f"{model} IS NOT A TUNED DEEPAR, PATCHTST OR FINE-TUNED CHRONOS-2 ENTRY")
    sensors = set(rows["sensor"])
    if pooled:
        kinds = {TASKS[sensor].kind for sensor in sensors}
        sensors = {sensor for sensor, task in TASKS.items() if task.kind in kinds}
    print(f"{model}|{covariates}|{int(pooled)}|{' '.join(sorted(sensors))}")
EOF
)
NSEEDS=$(wc -w <<< "$SEEDS")
echo "$(wc -l <<< "$PLANNED") PICKED CONFIGS x $NSEEDS SEEDS = $(( $(wc -l <<< "$PLANNED") * NSEEDS )) RUNS PLANNED (picks: $PICKS)"
echo "$PLANNED"
[ "${PLAN_ONLY:-0}" = 1 ] && exit 0

if grep -q "^chronos2_ft" <<< "$PLANNED"; then
    MIN_FREE_MIB=${MIN_FREE_MIB:-30000}
else
    MIN_FREE_MIB=${MIN_FREE_MIB:-8000}
fi
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
# The pinned Chronos-2 weights are in the deployed folder's cache; a fresh ssh shell has no HF_HOME.
export HF_HOME=${HF_HOME:-/scratch/ruha6285/sensbee_nvp/hf_cache}
export TMPDIR=/scratch/ruha6285/tmp  # fine-tuning checkpoints (~478 MB each, deleted after the fit)
mkdir -p "$TMPDIR"
echo "$(date -u +%FT%TZ) GPU $GPU ($FREE MiB free), commit $(cat COMMIT)"
# Without CUDA, the models would quietly train on the CPU.
venv-bench/bin/python -c "import sys, torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); sys.exit(not torch.cuda.is_available())" \
    || { echo "TORCH SEES NO GPU: stopping"; exit 1; }

# One harness run; prints a RUN line with the run time and the run folder, and stops the job when the run has a fit
# error or a failed origin. The harness output goes to the job's log through a copy of its stderr descriptor (on Linux,
# tee /dev/stderr would reopen the log file and overwrite it).
run_one() {
    local model=$1 set=$2 pooled=$3 sensors=$4 seed=$5
    # shellcheck disable=SC2206
    local args=(--snapshot "$DATA/snapshots/sensbee-2026-10-01" --out "$DATA/results" --model "$model"
                --period test --seed "$seed" --sensors $sensors)
    [ -n "$set" ] && args+=(--covariates "$set")
    [ "$pooled" = 1 ] && args+=(--pooled)
    local started=$SECONDS out
    out=$(venv-bench/bin/python scripts/run_harness.py "${args[@]}" 2>&1 | tee >(cat >&2) \
          | grep "^RUN WRITTEN to " | tail -1)
    echo "RUN $model ${set:-values} pooled=$pooled seed=$seed $((SECONDS - started))s ${out#RUN WRITTEN to }"
    venv-bench/bin/python - "${out#RUN WRITTEN to }" <<'EOF'
import json, sys
counts = json.load(open(sys.argv[1] + "/run.json"))["counts"]
failed = {s: sum(c["failed"] for c in n["contexts"].values()) for s, n in counts.items()}
errors = {s: n["fit"]["error"] for s, n in counts.items() if "error" in n.get("fit", {})}
if any(failed.values()) or errors:
    sys.exit(f"RUN HAS FAILURES: failed origins {failed}, fit errors {errors}")
EOF
}

while IFS='|' read -r model set pooled sensors; do
    for seed in $SEEDS; do
        run_one "$model" "$set" "$pooled" "$sensors" "$seed" < /dev/null
    done
done <<< "$PLANNED"
echo "$(date -u +%FT%TZ) ALL $(( $(wc -l <<< "$PLANNED") * NSEEDS )) RUNS DONE"
