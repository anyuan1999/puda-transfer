#!/bin/bash
# PUDA Cross-Domain Transfer: flash × {S1, S2}
# Single-GPU (GPU0) — runs in parallel with run_orthrus_s1_s2.sh on GPU1.
#
# Usage:
#   nohup bash run_flash_s1_s2.sh > artifacts/run_flash_s1_s2.log 2>&1 &
#
# Resume safe: PIDSMaker auto-skips already-completed stages via cfg-hash.

cd /root/PIDSMaker
source /root/miniconda3/etc/profile.d/conda.sh
conda activate pids
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0

DB=(--database_host 127.0.0.1 --database_port 5432 --database_user pgsql --database_password "")
ART=(--artifact_dir /root/PIDSMaker/artifacts)

# S1 = TRACE_E3 -> CADETS_E3   (Linux -> FreeBSD)
# S2 = CADETS_E3 -> TRACE_E3   (FreeBSD -> Linux)
SOURCES=(TRACE_E3 CADETS_E3)
TARGETS=(CADETS_E3 TRACE_E3)
SCENARIOS=(S1 S2)

DET=flash

RESULTS=/root/PIDSMaker/artifacts/transfer_results/all_results.csv
TASK_LOG_DIR=/root/PIDSMaker/artifacts/task_logs
mkdir -p /root/PIDSMaker/artifacts/transfer_results "$TASK_LOG_DIR"

echo "================================================================"
echo "# RUN  $DET × {S1, S2}   start: $(date)"
echo "# pid: $$  host: $(hostname)"
echo "# GPU: $CUDA_VISIBLE_DEVICES  (parallel with orthrus on GPU 1)"
echo "================================================================"

# ----------------------------------------------------------------------
# Phase 1: train flash on TRACE_E3
# ----------------------------------------------------------------------
echo
echo "===== PHASE 1: train $DET on TRACE_E3 (GPU $CUDA_VISIBLE_DEVICES) ====="
LOG="$TASK_LOG_DIR/train_${DET}_TRACE_E3.log"
echo "[TRAIN start $(date '+%H:%M:%S')] $DET on TRACE_E3 -> $LOG"
python pidsmaker/main.py "$DET" TRACE_E3 "${DB[@]}" "${ART[@]}" > "$LOG" 2>&1
RC=$?
if [ $RC -eq 0 ]; then
    echo "[TRAIN done  $(date '+%H:%M:%S')] $DET on TRACE_E3 OK"
else
    echo "[TRAIN FAIL  $(date '+%H:%M:%S')] $DET on TRACE_E3 rc=$RC"
    tail -20 "$LOG" | sed 's/^/    /'
    echo "Exiting due to train failure."
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
echo "# RUN flash×{S1,S2}  done: $(date)"
echo "# results CSV (S1+S2 rows for flash):"
grep -E "^flash,(S1|S2)," "$RESULTS"
echo "================================================================"
