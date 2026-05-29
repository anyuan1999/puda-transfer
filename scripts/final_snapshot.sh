#!/bin/bash
# Run when all tasks done OR before container deadline
PERSIST=/jizhicfs/PIDSMaker_root
ts=$(date +%Y%m%d_%H%M%S)
echo "[final_snapshot] $ts START"

# 1. Final rsync
rsync -a --delete-after /root/PIDSMaker/artifacts/ "$PERSIST/artifacts/"

# 2. Tar all critical files for portability
cd /
tar --exclude='PIDSMaker/data/*.dump' \
    --exclude='PIDSMaker/.git' \
    -czf "$PERSIST/full_pidsmaker_${ts}.tar.gz" \
    root/PIDSMaker/pidsmaker \
    root/PIDSMaker/config \
    root/PIDSMaker/scripts \
    root/PIDSMaker/scripts_persist \
    root/PIDSMaker/postgres \
    root/PIDSMaker/Ground_Truth \
    root/PIDSMaker/run_all_transfer_experiments.sh \
    root/PIDSMaker/run_orthrus_only.sh \
    root/PIDSMaker/transfer_inference.py \
    root/PIDSMaker/check_progress.sh \
    root/PIDSMaker/Dockerfile \
    root/PIDSMaker/pyproject.toml \
    root/PIDSMaker/entrypoint.sh \
    root/PIDSMaker/compose-pidsmaker.yml \
    root/PIDSMaker/compose-postgres.yml \
    root/PIDSMaker/download_datasets.sh \
    root/puda-transfer 2>&1 | tail -3

# 3. Copy results CSV outside artifacts for easy retrieval
cp /root/PIDSMaker/artifacts/transfer_results/all_results.csv "$PERSIST/all_results_${ts}.csv" 2>/dev/null

ls -lh "$PERSIST/" | head -10
echo "[final_snapshot] DONE  see $PERSIST"
