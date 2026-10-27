# HPC Cluster README

## Architecture

- **Login node (makalu48):** Has internet access. Downloads model weights from Hugging Face into shared `/scratch` storage.
- **GPU node (makalu86):** 4× NVIDIA A100 (40 GB VRAM each), no internet access. Sets `HF_HUB_OFFLINE=1` so transformers loads weights from disk.

Both nodes share the `/scratch` filesystem, so weights downloaded once on the login node are available on the GPU node without copying.

A 7B model requires approximately 14 GB in `float16` and fits within one A100. The benchmark uses multi-GPU inference: a VRAM check (`_get_usable_gpus`) selects GPUs with at least 15 GiB free, and forecasts are distributed across available GPUs using `ThreadPoolExecutor`.

### Deployment steps

```bash
# Deploy from your device
From sensbee_nvp, run this: ./hpc_setup/deploy_to_cluster.sh # This copies all necessary files to `/scratch/ruha6285/sensbee_nvp/` on the cluster.
# If it says permission denied, execute this: chmod +x hpc_setup/deploy_to_cluster.sh then try again
# to copy specific file run this: rsync -avz data/real_sensbee_json_data.json \ 
# ruha6285@cslogin.tu-ilmenau.de:/scratch/ruha6285/sensbee_nvp/data/

# Setup on cluster
ssh ruha6285@141.24.193.48  # or ssh ruha6285@cslogin.tu-ilmenau.de. Uses makalu48 for setup
From cd /scratch/ruha6285/sensbee_nvp or cd /scratch/$USER/sensbee_nvp, run this: ./hpc_setup/setup_cluster.sh

# Download the models beforehand
source /scratch/$USER/sensbee_nvp/env_setup.sh
cd /scratch/ruha6285/sensbee_nvp
python hpc_setup/download_models.py --all # or --model mistral-7b

# Gated models (Llama 2): re-authenticate on the cluster
# Your Hugging Face login on your laptop does not apply on the cluster. Do one of the following on the login node:
#   Option A — interactive login (stores token in ~/.cache/huggingface/token on the cluster):
#     pip install -q huggingface_hub && huggingface-cli login
#     Then paste your token from https://huggingface.co/settings/tokens (and accept the Llama 2 license on HF if needed).
#   Option B — use a token in the environment (no interactive step):
#     export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
#     python hpc_setup/download_models.py --model llama2-7b
# Do not commit HF_TOKEN to git.

# Test on GPU node
ssh -X makalu86 # after this you can also run nvidia-smi to verify GPU access
cd /scratch/ruha6285/sensbee_nvp
export HF_HUB_OFFLINE=1 # maybe to make gpu offline. 
source /scratch/$USER/sensbee_nvp/env_setup.sh

# Testing
python hpc_setup/test_forecast.py --source synthetic --model mistral-7b # --model llama2-7b

# Test with local JSON file
python hpc_setup/test_forecast.py --source json \
    --json-path data/real_sensbee_json_data.json \
    --column temperature
    --model llama2-7b

python hpc_setup/test_forecast.py --source api --sensor-id <UUID> --column temperature
```

### From SensBee API

```bash
python3 hpc_setup/run_custom_forecast.py \
  --source api \
  --sensor-id "YOUR_SENSOR_UUID" \
  --api-key "YOUR_SENSBEE_READ_API_KEY" \
  --column temperature \
  --model mistral-7b \
  --horizon 48 \
  --history-days 7 \
  --samples 10 \
  --output-dir results/
```

### Storage Management

# Check cache size:
```bash
du -sh /scratch/ruha6285/.cache/huggingface/
```

# Storage locations:
- `/scratch/ruha6285/sensbee_nvp/` - Project files (2TB available)
- `/scratch/ruha6285/.cache/huggingface/` - Model cache (~14GB per 7B model)
- `/data2/hpc-user/ruha6285/` - Backup storage (100GB, backed up)

# Clean cache (if needed):
```bash
rm -rf /scratch/ruha6285/.cache/huggingface/hub/
```

## Running the Benchmark

The benchmark script (`run_benchmark.py`) runs a systematic evaluation across sensors, window sizes, and ablation configurations:

```bash
# On GPU node (makalu86)
cd /scratch/ruha6285/sensbee_nvp
source /scratch/$USER/sensbee_nvp/env_setup.sh
export HF_HUB_OFFLINE=1

# Full ablation study (8 configs × 3 windows × 2 sensors = 48 tests)
python hpc_setup/run_benchmark.py --provider local --model llama2-7b --mode ablation

# Quick smoke test (1 window, 1 sensor)
python hpc_setup/run_benchmark.py --provider local --model mistral-7b --quick

# Temperature only
python hpc_setup/run_benchmark.py --provider local --model llama2-7b --sensor temperature

# Groq API benchmark (requires GROQ_API_KEY)
export GROQ_API_KEY='...'
python hpc_setup/run_benchmark.py --provider groq --mode ablation
```

Results are saved to `results/report_benchmark_<timestamp>.json` with figures in `results/figures/` and LaTeX tables in `results/tables/`.

### Benchmark data (run_benchmark.py)

The benchmark expects these files under `data/`:

- **Temperature:** `data/temp_14day_sensbee_data.json`
- **Visitors:** `data/eishalle_14day_sensbee_data.json`

`./hpc_setup/deploy_to_cluster.sh` copies `data/temp_*.json` and `data/eishalle_*.json` from your machine if present. If you see **"Data file NOT FOUND: .../data/temp_1..."** (path truncated), the full path is `.../data/temp_14day_sensbee_data.json` — either redeploy (so data/ is included) or copy data manually:

```bash
rsync -avz data/temp_14day_sensbee_data.json data/eishalle_14day_sensbee_data.json \
  ruha6285@cslogin.tu-ilmenau.de:/scratch/ruha6285/sensbee_nvp/data/
```

### Clear cache, re-download models, and remove benchmark results

**On the cluster** (login node for cache/download; any node for results):

1. **Clear HuggingFace cache** (so models are re-downloaded next time):
   ```bash
   rm -rf /scratch/ruha6285/sensbee_nvp/hf_cache/*
   # If you also use the default cache location:
   # rm -rf /scratch/ruha6285/.cache/huggingface/hub/
   ```

2. **Re-download all models** (run on login node with internet, after clearing cache):
   ```bash
   cd /scratch/ruha6285/sensbee_nvp
   source /scratch/$USER/sensbee_nvp/env_setup.sh
   python hpc_setup/download_models.py --all
   ```

3. **Remove all benchmark result JSONs** (on cluster):
   ```bash
   rm -f /scratch/ruha6285/sensbee_nvp/results/benchmark_*.json
   ```
   To list before deleting: `ls /scratch/ruha6285/sensbee_nvp/results/benchmark_*.json`
## Benchmark harness on the cluster

The harness (`scripts/run_harness.py`) runs from the same deployed copy, with its own Python 3.12 environment (`venv-bench`, step 3b). Every command runs from your local machine or on the cluster as noted; nothing here starts a job by itself.

1. **Check the cluster (read-only)**, from your local machine with the VPN on, and keep the output:
   ```bash
   ssh ruha6285@cslogin.tu-ilmenau.de 'bash -s' < hpc_setup/check_cluster.sh
   ssh -J ruha6285@cslogin.tu-ilmenau.de ruha6285@makalu86 'bash -s' < hpc_setup/check_cluster.sh
   ```
   It prints the Python versions and modules, GPUs and driver, scheduler, free space on `/scratch` and whether huggingface.co is reachable. For the GPU node, `-J` (jump host) asks for each password on your terminal; a nested `ssh` inside the first command would read it from the piped script and fail.
   Result 2026-10-26 (login node makalu47): Python 3.12.15 and 3.11 (no 3.13), no module system, no scheduler, an RTX A5000 (24 GB, shared), driver 595.91.07 (CUDA 13.2), 57 TB free on `/scratch`, huggingface.co reachable.
2. **Deploy the code** from a clean checkout: `./hpc_setup/deploy_to_cluster.sh`. It also copies `tests/`, `pytest.ini` and the requirement and constraint files, and writes `COMMIT` (the deployed commit and whether the checkout was dirty), which every `run.json` records.
3. **Copy a snapshot** (private data; keep it on `/scratch`, never in a shared folder):
   ```bash
   ssh ruha6285@cslogin.tu-ilmenau.de 'mkdir -p /scratch/ruha6285/benchmark_data/snapshots'
   rsync -av ../../benchmark_data/snapshots/sensbee-2026-10-01 ruha6285@cslogin.tu-ilmenau.de:/scratch/ruha6285/benchmark_data/snapshots/
   ```
   The harness checks each file's SHA-256 against the manifest on load.
3b. **Environment** on the login node (Python 3.12, the Mac's pins): `bash hpc_setup/setup_bench_cluster.sh`. It ends with a smoke import and freezes the result to `constraints-bench-cluster.txt`.
4. **Pinned weights** on the login node, into `$HF_HOME`: `python hpc_setup/download_models.py --pinned chronos2 mistral7b llama2_7b` (Llama 2 needs its licence accepted on Hugging Face and `huggingface-cli login` first). Each file's SHA-256 is checked against the Hub.
5. **Runs** on the GPU node with `HF_HUB_OFFLINE=1` and the same `scripts/run_harness.py` commands as on the Mac.
   GPU node check 2026-10-26 (makalu86): 4× A100-PCIE-40GB shared with other users, no scheduler, Python 3.12.15 (`python3` is 3.6.8), no internet.
6. **Job scripts** (`hpc_setup/jobs/`): each takes the GPU with the most free memory, refuses to start below a minimum, and stops at the first failed run. There is no queue, so start them under `nohup` from the GPU node; they keep running when the ssh session drops:
   ```bash
   ssh -J ruha6285@cslogin.tu-ilmenau.de ruha6285@makalu86
   cd /scratch/ruha6285/sensbee_nvp && mkdir -p /scratch/ruha6285/benchmark_data/logs
   GATE=1 nohup bash hpc_setup/jobs/tuned_dl_validation.sh /scratch/ruha6285/benchmark_data deepar > /scratch/ruha6285/benchmark_data/logs/deepar_gate.log 2>&1 &
   ```
   `tail -f` the log to follow it. `GATE=1` runs only the first run, twice (timing projection and same-seed check), and stops; the same command without `GATE=1` runs the full grid.
   Fine-tuned Chronos-2 (6a.7): `hpc_setup/jobs/finetune_validation.sh /scratch/ruha6285/benchmark_data` (no family argument; needs 30 GB free on one GPU; its `GATE=1` also runs gate Z against the Mac zero-shot run `2026-10-14T222254Z_chronos2`, which must be in `/scratch/ruha6285/benchmark_data/results/`).
