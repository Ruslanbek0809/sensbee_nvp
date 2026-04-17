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