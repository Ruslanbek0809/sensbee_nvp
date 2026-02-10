#!/bin/bash
# Deployment script. You can run this on your LOCAL machine to copy files to cluster
# Usage: ./hpc_setup/deploy_to_cluster.sh

set -e

# Configuration
USERNAME="ruha6285"
CLUSTER_HOST="cslogin.tu-ilmenau.de"
PROJECT_NAME="sensbee_nvp"
REMOTE_DIR="/scratch/$USERNAME/$PROJECT_NAME"

# Local project directory (parent of hpc_setup)
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "Local directory: $LOCAL_DIR"
echo "Remote directory: $REMOTE_DIR"
echo "Cluster: $USERNAME@$CLUSTER_HOST"
echo ""

# Files and directories to copy
echo "Preparing files for deployment..."

# Create temporary directory for deployment
TEMP_DIR=$(mktemp -d)
echo "Temporary directory: $TEMP_DIR"

# Copy necessary files
cp -r "$LOCAL_DIR/src" "$TEMP_DIR/"
cp -r "$LOCAL_DIR/hpc_setup" "$TEMP_DIR/"
cp "$LOCAL_DIR/requirements.txt" "$TEMP_DIR/" 2>/dev/null || true

# Create remote directory
echo ""
echo "Creating remote directory structure..."
ssh $USERNAME@$CLUSTER_HOST "mkdir -p $REMOTE_DIR"

# Copy files to cluster
echo ""
echo "Copying files to cluster..."
rsync -avz --progress \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.git' \
    --exclude='*.egg-info' \
    --exclude='venv' \
    --exclude='llm_env' \
    "$TEMP_DIR/" $USERNAME@$CLUSTER_HOST:$REMOTE_DIR/

# Make scripts executable
echo ""
echo "Making scripts executable..."
ssh $USERNAME@$CLUSTER_HOST "chmod +x $REMOTE_DIR/hpc_setup/*.sh"

# Cleanup
rm -rf "$TEMP_DIR"

echo ""
echo "DEPLOYMENT COMPLETE!"
echo ""
