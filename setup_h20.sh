#!/bin/bash
# One-click setup for H20 dual-GPU machine
set -e

echo "=== Setting up PIDSMaker + PUDA Transfer on H20 ==="

# 1. Clone PIDSMaker
if [ ! -d "PIDSMaker" ]; then
    git clone https://github.com/ubc-provenance/PIDSMaker.git
    cd PIDSMaker
    git checkout tags/2.1.0 -b v2.1.0
else
    cd PIDSMaker
fi

# 2. Apply patches
echo "Applying patches..."
git apply ../patches/enable_model_saving.patch 2>/dev/null || echo "Patch 1 already applied or conflict"
git apply ../patches/fix_save_model.patch 2>/dev/null || echo "Patch 2 already applied or conflict"

# 3. Copy transfer scripts
cp ../transfer_inference.py .
cp ../run_all_transfer_experiments.sh .

# 4. Setup env
cp .env.local .env
mkdir -p data artifacts

# 5. Modify Dockerfile for H20 CUDA 12.4
# (You may need to manually edit Dockerfile with patches/Dockerfile.h20.patch)

echo ""
echo "=== Setup Complete ==="
echo "Next steps:"
echo "  1. Edit Dockerfile to use CUDA 12.4 PyTorch (see patches/Dockerfile.h20.patch)"
echo "  2. docker compose -p postgres -f compose-postgres.yml up -d --build"
echo "  3. docker compose -f compose-pidsmaker.yml up -d --build"
echo "  4. Download datasets and load into PostgreSQL"
echo "  5. Run: bash run_all_transfer_experiments.sh"
