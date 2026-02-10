#!/bin/bash
# HPC Cluster Setup Script for SensBee NVP Local LLM. There are 2 nodes: makalu48 (setup. Has internet, download models here) and makalu86 (GPU node. Has GPU, run forecasts here (may lack internet)).

set -e  # Exit on error

# Configuration
PROJECT_NAME="sensbee_nvp"
VENV_NAME="llm_env"
WORK_DIR="${SCRATCH:-/scratch/$USER}/$PROJECT_NAME"
HF_CACHE_DIR="$WORK_DIR/hf_cache"

# Step 1: Create directories
echo ""
echo "Step 1: Creating directories..."
mkdir -p "$WORK_DIR"/{data,results,logs,hf_cache}
cd "$WORK_DIR"
echo "Work directory: $WORK_DIR"

# Step 2: Find suitable Python version
echo ""
echo "Step 2: Detecting Python..."
if command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
elif command -v python3.10 &> /dev/null; then
    PYTHON_CMD="python3.10"
elif command -v python3.9 &> /dev/null; then
    PYTHON_CMD="python3.9"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    echo "ERROR: No suitable Python found!"
    exit 1
fi

echo "Using: $PYTHON_CMD ($($PYTHON_CMD --version))"

# Step 3: Create virtual environment
echo ""
echo "Step 3: Creating virtual environment..."
if [ ! -d "$VENV_NAME" ]; then
    $PYTHON_CMD -m venv "$VENV_NAME"
    echo "✓ Created: $WORK_DIR/$VENV_NAME"
else
    echo "✓ Already exists: $WORK_DIR/$VENV_NAME"
fi

# Step 4: Activate virtual environment
echo ""
echo "Step 4: Activating virtual environment..."
source "$VENV_NAME/bin/activate"
echo "Active Python: $(which python)"

# Step 5: Upgrade pip
echo ""
echo "Step 5: Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Step 6: Install PyTorch with CUDA
echo ""
echo "Step 6: Installing PyTorch with CUDA 11.8 support..."
echo "   This may take 5-10 minutes..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Step 7: Install transformers ecosystem
echo ""
echo "Step 7: Installing transformers and accelerate..."
pip install \
    transformers>=4.35.0 \
    accelerate>=0.24.0 \
    sentencepiece>=0.1.99 \
    protobuf>=3.20.0 \
    safetensors>=0.4.0

# Step 8: Install bitsandbytes (optional, for quantization)
echo ""
echo "Step 8: Installing bitsandbytes for quantization..."
pip install bitsandbytes>=0.41.0 || {
    echo "⚠ bitsandbytes installation failed (optional)"
}

# Step 9: Install project dependencies
echo ""
echo "Step 9: Installing project dependencies..."
pip install \
    pandas>=2.1.0 \
    numpy>=1.24.0 \
    pyyaml>=6.0.0 \
    python-dotenv>=1.0.0 \
    httpx>=0.25.0 \
    scipy>=1.11.0

# Step 10: Copy project files if source directory exists
echo ""
echo "Step 10: Setting up project files..."

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$(dirname "$SCRIPT_DIR")"

# Check if source and work dir are different
if [ "$(realpath "$SOURCE_DIR")" = "$(realpath "$WORK_DIR")" ]; then
    echo "✓ Project files already in place"
elif [ -d "$SOURCE_DIR/src" ]; then
    echo "Copying project files from: $SOURCE_DIR"
    
    # Copy source code
    cp -r "$SOURCE_DIR/src" "$WORK_DIR/"
    
    # Copy hpc_setup scripts
    cp -r "$SOURCE_DIR/hpc_setup" "$WORK_DIR/"
    
    # Copy data if exists
    if [ -d "$SOURCE_DIR/data" ]; then
        cp -r "$SOURCE_DIR/data/"* "$WORK_DIR/data/" 2>/dev/null || true
    fi
    
    # Copy requirements if exists
    if [ -f "$SOURCE_DIR/requirements.txt" ]; then
        cp "$SOURCE_DIR/requirements.txt" "$WORK_DIR/"
    fi
    
    echo "✓ Project files copied"
else
    echo "⚠ Source directory not found. Make sure to copy project files manually."
fi

# Step 11: Set up environment variables
echo ""
echo "Step 11: Setting up environment variables..."
cat > "$WORK_DIR/env_setup.sh" << 'EOF'
#!/bin/bash
# Environment setup for SensBee NVP
export WORK_DIR="${SCRATCH:-/scratch/$USER}/sensbee_nvp"
export HF_HOME="$WORK_DIR/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"

# Set offline mode for GPU nodes without internet
# Uncomment the following lines when running on GPU nodes:
# export HF_HUB_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1

# Activate virtual environment
source "$WORK_DIR/llm_env/bin/activate"

echo "Environment ready:"
echo "  WORK_DIR: $WORK_DIR"
echo "  HF_HOME: $HF_HOME"
echo "  Python: $(which python)"
EOF
chmod +x "$WORK_DIR/env_setup.sh"
echo "✓ Created env_setup.sh"

# Step 12: Verify installation
echo ""
echo "Step 12: Verifying installation..."
python -c "import torch; print(f'✓ PyTorch: {torch.__version__}')"
python -c "import torch; print(f'✓ CUDA available: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'✓ Transformers: {transformers.__version__}')"
python -c "import pandas; print(f'✓ Pandas: {pandas.__version__}')"
python -c "import numpy; print(f'✓ NumPy: {numpy.__version__}')"

# Summary
echo ""
echo "✓ SETUP COMPLETE!"
echo ""
echo "Work directory: $WORK_DIR"
echo "Virtual environment: $WORK_DIR/$VENV_NAME"
echo "HuggingFace cache: $HF_CACHE_DIR"
echo ""
echo "NEXT STEPS (on login node with internet):"
echo ""
echo "1. Activate environment:"
echo "   source $WORK_DIR/env_setup.sh"
echo ""
echo "2. Download models (required, ~5GB for mistral-7b):"
echo "   cd $WORK_DIR"
echo "   python hpc_setup/download_models.py --model mistral-7b"
echo ""
echo "3. Prepare offline data (if fetching from API):"
echo "   python hpc_setup/prepare_offline_data.py"
echo ""
echo "RUNNING ON GPU NODE:"
echo ""
echo "Interactive session"
echo "   ssh makalu86  # or srun --gres=gpu:1 --pty bash"
echo "   source $WORK_DIR/env_setup.sh"
echo "   export HF_HUB_OFFLINE=1"
echo "   python hpc_setup/test_local_models.py --model mistral-7b"
