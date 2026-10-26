#!/bin/bash
# Gate E (Phase 6a): is the cluster environment equivalent to the Mac's? Runs zero-shot Chronos-2 on the validation
# period (5 main sensors, contexts 1/7/28 d; the adapter runs on the CPU) and compares its forecasts with the Mac run
# 2026-10-14T222254Z_chronos2 (same snapshot, period, contexts). Pass: every quantile within 1e-3 relative.
# GPU-hours: 0 (CPU only); wall time a few minutes.
# Usage, on the cluster from the deployed project folder (the Mac run's folder copied to the cluster first):
#   bash hpc_setup/jobs/gate_e_chronos2.sh /scratch/ruha6285/benchmark_data <Mac run folder on the cluster>
set -e
if [ $# -ne 2 ] || [ ! -f "$1/snapshots/sensbee-2026-10-01/manifest.json" ] || [ ! -f "$2/run.json" ]; then
    echo "USAGE (on the cluster): bash hpc_setup/jobs/gate_e_chronos2.sh <benchmark_data folder> <Mac run folder>"
    echo "  the benchmark_data folder must hold snapshots/sensbee-2026-10-01, the Mac run folder its run.json"
    exit 1
fi
cd "$(dirname "$0")/../.."
DATA=$1
MAC_RUN=$2
export HF_HUB_OFFLINE=1
venv-bench/bin/python scripts/run_harness.py --snapshot "$DATA/snapshots/sensbee-2026-10-01" \
    --out "$DATA/results" --model chronos2 --period validation --contexts 1 7 28 \
    --sensors MANEBACH_WEATHER_STATION STUTZERBACH_WEATHER_STATION FRAUENWALD_WEATHER_STATION EISHALLE SCHWIMMHALLE
CLUSTER_RUN=$(ls -d "$DATA"/results/*_chronos2 | tail -1)
venv-bench/bin/python scripts/compare_runs.py "$MAC_RUN" "$CLUSTER_RUN" --rtol 1e-3
