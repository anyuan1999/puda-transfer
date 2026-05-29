#!/bin/bash
# Wait until 1 hour before deadline, take snapshot, optionally graceful stop
DEADLINE_TS=$(date -d '2026-05-29 10:00:00' +%s)  # 1h before container expires
while true; do
    NOW=$(date +%s)
    REMAIN=$((DEADLINE_TS - NOW))
    if [ $REMAIN -le 0 ]; then
        echo "[$(date)] Deadline reached -- taking final snapshot..."
        bash /root/PIDSMaker/scripts_persist/final_snapshot.sh
        echo "[$(date)] Snapshot done. Tasks still running will continue but next sync covers them."
        # Loop further snapshots every 15 min until 11:00
        for i in 1 2 3 4; do
            sleep 900
            bash /root/PIDSMaker/scripts_persist/final_snapshot.sh
        done
        exit 0
    fi
    sleep 60
done
