# HPC Cluster README

Nodes: makalu48 (setup. Has internet), makalu86 (GPU node. No internet)

<!-- ## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ makalu48 (Login Node)            makalu86 (GPU Node)            │
│ ✓ Internet access                ✓ 4x NVIDIA A100 (160GB)       │
│ ✓ Fetch from SensBee API         ✗ No internet                  │
│ ✓ Run main.py service            ✓ Run inference_service.py     │
│                                                                  │
│  User Request ──► main.py ──────► inference_service.py          │
│                   (fetch data)    (run local LLM)               │
│                       ◄──────────────────┘                      │
│                   (return forecast)                              │
└─────────────────────────────────────────────────────────────────┘
``` -->

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
  --sensor-id "8d790e21-f948-4e75-9c47-ea8b1aa75e9d" \
  --api-key "6d5ecc8d-1e5a-4c66-be1d-c6fd34958777" \
  --column humidity \
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


<!-- ## Production Deployment

### Option 1: Two-Node Service (Recommended)

**On makalu86 (GPU node) - Start inference service:**
```bash
source /scratch/$USER/sensbee_nvp/env_setup.sh
export HF_HUB_OFFLINE=1
uvicorn src.service.inference_service:app --host 0.0.0.0 --port 8001
```

**On makalu48 (login node) - Start main service:**
```bash
source /scratch/$USER/sensbee_nvp/env_setup.sh
export INFERENCE_SERVICE_URL=http://makalu86:8001
export SENSBEE_API_KEY="your-api-key"
uvicorn src.service.main:app --host 0.0.0.0 --port 8000
```

**User makes request to makalu48:**
```bash
curl -X POST http://makalu48:8000/forecast \
    -H "Content-Type: application/json" \
    -d '{
        "sensor_id": "8d790e21-f948-4e75-9c47-ea8b1aa75e9d",
        "column": "temperature",
        "horizon_hours": 24,
        "provider": "remote"
    }'
``` -->

<!-- ### Option 2: Batch Processing

For one-off forecasts without running services:

```bash
# On makalu48 (fetch data)
python hpc_setup/production_forecast.py --fetch-only \
    --sensor-id 8d790e21-f948-4e75-9c47-ea8b1aa75e9d \
    --column temperature \
    --output-data /scratch/$USER/sensbee_nvp/data/request.json

# On makalu86 (generate forecast)
python hpc_setup/production_forecast.py --from-file \
    --input-data /scratch/$USER/sensbee_nvp/data/request.json \
    --model mistral-7b
``` -->

