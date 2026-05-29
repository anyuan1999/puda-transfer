#!/bin/bash
# Continuous incremental sync of artifacts + code patches to persistent CFS
PERSIST=/jizhicfs/PIDSMaker_root
mkdir -p "$PERSIST/artifacts" "$PERSIST/logs"

while true; do
    ts=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$ts] sync start" >> "$PERSIST/logs/sync.log"
    # incremental rsync: artifacts (only changed files)
    rsync -a --delete-after \
        /root/PIDSMaker/artifacts/ "$PERSIST/artifacts/" \
        >> "$PERSIST/logs/sync.log" 2>&1
    # also copy patched source files (small, idempotent)
    rsync -a \
        /root/PIDSMaker/run_all_transfer_experiments.sh \
        /root/PIDSMaker/run_orthrus_only.sh \
        /root/PIDSMaker/transfer_inference.py \
        /root/PIDSMaker/check_progress.sh \
        /root/PIDSMaker/scripts_persist/ \
        /root/PIDSMaker/pidsmaker/detection/training_methods/training_loop.py \
        /root/PIDSMaker/pidsmaker/utils/data_utils.py \
        /root/PIDSMaker/pidsmaker/featurization/feat_inference_methods/feat_inference_flash.py \
        /root/PIDSMaker/config/orthrus.yml \
        "$PERSIST/code_files/" \
        >> "$PERSIST/logs/sync.log" 2>&1
    echo "[$(date '+%H:%M:%S')] sync done, artifacts size: $(du -sh "$PERSIST/artifacts" | cut -f1)" >> "$PERSIST/logs/sync.log"
    sleep 300  # 5 min
done
