#!/bin/bash
# Builds the benchmark environment (venv-bench) on the cluster, on the login node (it has internet; the GPU node
# doesn't). /scratch is shared, so the GPU node uses the same venv.
# The cluster has no Python 3.13 (check C0, 2026-10-26: 3.12.15 and 3.11 only), so the venv uses Python 3.12 with the
# Mac's pinned versions (constraints-bench.txt). PyPI's Linux torch 2.10.0 wheel is a CUDA build.
# The cluster run of zero-shot Chronos-2 is compared with the Mac run (gate E) before any cluster result counts.
# Usage, on the login node, from the deployed project folder:
#   bash hpc_setup/setup_bench_cluster.sh
set -e
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-/usr/bin/python3.12}
if [ ! -x "$PYTHON" ]; then
    echo "$PYTHON NOT FOUND: run this on the cluster's login node (the Mac has its own venv-bench)"; exit 1
fi
if [ -d venv-bench ]; then
    echo "venv-bench EXISTS; remove it first to rebuild"; exit 1
fi
"$PYTHON" -m venv venv-bench
venv-bench/bin/pip install --upgrade pip
venv-bench/bin/pip install -r requirements-bench.txt
venv-bench/bin/pip check
# Smoke import of the top-level packages (pip check passing isn't enough, AGENTS.md "Gotchas").
venv-bench/bin/python -c "
import autogluon.timeseries, chronos, timesfm, tirex, statsforecast, torch, transformers, peft, scipy, lightgbm
print('IMPORTS OK; torch', torch.__version__, 'built for CUDA', torch.version.cuda)"
venv-bench/bin/pip freeze > constraints-bench-cluster.txt
echo "FROZEN to constraints-bench-cluster.txt; compare it with constraints-bench.txt"
