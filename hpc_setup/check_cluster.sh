#!/bin/bash
# Read-only check of the cluster before the benchmark environment is set up (Phase 6a, check C0).
# It changes nothing: it prints the Python versions and modules on offer, the GPUs and driver, the scheduler, the
# free space on /scratch, and what an earlier deployment left there.
# Run it from your LOCAL machine (VPN on), and paste the output:
#   ssh ruha6285@cslogin.tu-ilmenau.de 'bash -s' < hpc_setup/check_cluster.sh
#   ssh ruha6285@cslogin.tu-ilmenau.de "ssh makalu86 'bash -s'" < hpc_setup/check_cluster.sh

echo "== host"; hostname; uname -sr
echo "== python on PATH"
for p in python3.13 python3.12 python3.11 python3 conda; do
    command -v "$p" >/dev/null 2>&1 && echo "$p: $($p --version 2>&1) ($(command -v $p))"
done
echo "== environment modules (python, cuda, gcc)"
# A non-interactive shell may inherit MODULESHOME without the module function: load it first.
if ! command -v module >/dev/null 2>&1 && [ -n "$MODULESHOME" ] && [ -f "$MODULESHOME/init/bash" ]; then
    . "$MODULESHOME/init/bash"
fi
if command -v module >/dev/null 2>&1; then
    module -t avail 2>&1 | grep -i -E "python|cuda|gcc|conda|anaconda" | head -40
else
    echo "no module command"
fi
echo "== GPUs"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv \
    || echo "no nvidia-smi on this node"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi | grep -i "CUDA Version"
echo "== scheduler"
for c in sinfo squeue sbatch qsub; do command -v "$c" >/dev/null 2>&1 && echo "$c: $(command -v $c)"; done
command -v sinfo >/dev/null 2>&1 && sinfo -s
command -v squeue >/dev/null 2>&1 && squeue -u "$USER"
echo "== scratch"
df -h "/scratch/$USER" 2>&1 | tail -1
ls -la "/scratch/$USER" 2>&1 | head -20
ls "/scratch/$USER/sensbee_nvp" 2>&1 | head -20
echo "== internet (login node only)"
timeout 5 bash -c 'echo > /dev/tcp/huggingface.co/443' 2>/dev/null && echo "huggingface.co reachable" || echo "huggingface.co NOT reachable"
