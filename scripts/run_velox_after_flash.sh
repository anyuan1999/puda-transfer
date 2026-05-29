#!/bin/bash
# Watcher: wait for flash@GPU0 to finish, then auto-launch velox@GPU0.
#
# Triggered after flash's main.py PID exits.
# Resume safe: PIDSMaker auto-skips already-completed stages via cfg-hash;
# if container deadline hits before velox finishes training, intermediate
# epoch checkpoints are still saved to artifacts/ -> rsync'd to jizhicfs.
#
# Usage:
#   nohup bash run_velox_after_flash.sh > artifacts/run_velox_after_flash.log 2>&1 &

cd /root/PIDSMaker
source /root/miniconda3/etc/profile.d/conda.sh
conda activate pids
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8

DB=(--database_host 127.0.0.1 --database_port 5432 --database_user pgsql --database_password "")
ART=(--artifact_dir /root/PIDSMaker/artifacts)

SOURCES=(TRACE_E3 CADETS_E3)
TARGETS=(CADETS_E3 TRACE_E3)
SCENARIOS=(S1 S2)

DET=velox

RESULTS=/root/PIDSMaker/artifacts/transfer_results/all_results.csv
TASK_LOG_DIR=/root/PIDSMaker/artifacts/task_logs
mkdir -p /root/PIDSMaker/artifacts/transfer_results "$TASK_LOG_DIR"

echo "================================================================"
echo "# WATCHER  velox-after-flash   start: $(date)"
echo "# pid: $$  host: $(hostname)"
echo "================================================================"

# ----------------------------------------------------------------------
# Phase 0: wait for flash main.py to exit
# ----------------------------------------------------------------------
echo
echo "===== PHASE 0: wait for flash on TRACE_E3 to finish ====="
WAIT_LOOP=0
while pgrep -f "main\.py flash TRACE_E3" >/dev/null 2>&1; do
    WAIT_LOOP=$((WAIT_LOOP + 1))
    if [ $((WAIT_LOOP % 20)) -eq 1 ]; then
        # log every 10 min
        flash_log=/root/PIDSMaker/artifacts/task_logs/train_flash_TRACE_E3.log
        last_line=$(tail -1 "$flash_log" 2>/dev/null | cut -c1-100)
        echo "[$(date '+%H:%M:%S')] flash still training. last log: $last_line"
    fi
    sleep 30
done
echo "[$(date '+%H:%M:%S')] flash main.py exited."

# Check exit status by looking at run_flash_s1_s2.log
if grep -q "TRAIN done" /root/PIDSMaker/artifacts/run_flash_s1_s2.log 2>/dev/null; then
    echo "[$(date '+%H:%M:%S')] flash TRAIN reported success."
else
    echo "[$(date '+%H:%M:%S')] flash may have failed/aborted (no 'TRAIN done' marker). Continuing anyway -- velox is independent."
fi

# Wait 60s for GPU0 to drain & memory to be released
echo "[$(date '+%H:%M:%S')] cooling down 60s before launching velox..."
sleep 60
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv 2>&1

# ----------------------------------------------------------------------
# Phase 1: train velox on TRACE_E3 (GPU 0)
# ----------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES=0
echo
echo "===== PHASE 1: train $DET on TRACE_E3 (GPU $CUDA_VISIBLE_DEVICES) ====="
LOG="$TASK_LOG_DIR/train_${DET}_TRACE_E3.log"
echo "[TRAIN start $(date '+%H:%M:%S')] $DET on TRACE_E3 -> $LOG"
python pidsmaker/main.py "$DET" TRACE_E3 "${DB[@]}" "${ART[@]}" > "$LOG" 2>&1
RC=$?
if [ $RC -eq 0 ]; then
    echo "[TRAIN done  $(date '+%H:%M:%S')] $DET on TRACE_E3 OK"
else
    echo "[TRAIN FAIL  $(date '+%H:%M:%S')] $DET on TRACE_E3 rc=$RC (likely deadline-induced kill or OOM)"
    tail -20 "$LOG" | sed 's/^/    /'
    echo "Skipping Phase 2 transfer since train didn't finish cleanly."
    exit $RC
fi

# ----------------------------------------------------------------------
# Phase 2: transfer S1 + S2
# ----------------------------------------------------------------------
echo
echo "===== PHASE 2: transfer S1 + S2 ====="
for i in "${!SOURCES[@]}"; do
    src="${SOURCES[$i]}"
    tgt="${TARGETS[$i]}"
    sc="${SCENARIOS[$i]}"
    log="$TASK_LOG_DIR/transfer_${DET}_${sc}_${src}_to_${tgt}.log"

    echo "[TRANSFER start $(date '+%H:%M:%S')] [$sc] $DET $src->$tgt"
    python transfer_inference.py "$DET" "$src" "$tgt" "${ART[@]}" "${DB[@]}" > "$log" 2>&1
    rc=$?
    output=$(cat "$log")

    tp=$(echo  "$output" | grep "^.*- tp:"        | tail -1 | awk '{print $NF}')
    tn=$(echo  "$output" | grep "^.*- tn:"        | tail -1 | awk '{print $NF}')
    fp=$(echo  "$output" | grep "^.*- fp:"        | tail -1 | awk '{print $NF}')
    fn=$(echo  "$output" | grep "^.*- fn:"        | tail -1 | awk '{print $NF}')
    mcc=$(echo "$output" | grep "^.*- mcc:"       | tail -1 | awk '{print $NF}')
    f1=$(echo  "$output" | grep "^.*- fscore:"    | tail -1 | awk '{print $NF}')
    adp=$(echo "$output" | grep "^.*- adp_score:" | tail -1 | awk '{print $NF}')

    if [ -n "$tp" ]; then
        echo "[TRANSFER done  $(date '+%H:%M:%S')] [$sc] $DET $src->$tgt OK TP=$tp FN=$fn MCC=$mcc F1=$f1 ADP=$adp"
    else
        err=$(echo "$output" | grep -E "Error|Killed|Traceback" | tail -1 | cut -c1-120)
        echo "[TRANSFER FAIL  $(date '+%H:%M:%S')] [$sc] $DET $src->$tgt rc=$rc -- $err"
    fi
    echo "$DET,$sc,$src,$tgt,$tp,$tn,$fp,$fn,$mcc,$f1,$adp" >> "$RESULTS"
done

echo
echo "================================================================"
echo "# WATCHER velox×{S1,S2}  done: $(date)"
echo "# results CSV (S1+S2 rows for velox):"
grep -E "^velox,(S1|S2)," "$RESULTS"
echo "================================================================"
