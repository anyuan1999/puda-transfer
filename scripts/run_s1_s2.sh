#!/bin/bash
# PUDA Cross-Domain Transfer: S1 + S2 (Trace <-> CADETS_E3)
# Dual-GPU parallel scheduling:
#   GPU0: flash -> velox      (sequential on same GPU)
#   GPU1: orthrus             (alone)
#   Then 6 transfers, 3 per GPU
#
# Usage:
#   nohup bash run_s1_s2.sh > artifacts/run_s1_s2.log 2>&1 &
#
# Resume safe: PIDSMaker auto-skips train if cfg-hash artifacts already exist.

cd /root/PIDSMaker

# Activate conda env (must, since this is run via nohup -- inherits no env)
source /root/miniconda3/etc/profile.d/conda.sh
conda activate pids

export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8

DB=(--database_host 127.0.0.1 --database_port 5432 --database_user pgsql --database_password "")
ART=(--artifact_dir /root/PIDSMaker/artifacts)

# S1 = TRACE_E3 -> CADETS_E3   (Linux -> FreeBSD)
# S2 = CADETS_E3 -> TRACE_E3   (FreeBSD -> Linux)
SOURCES=(TRACE_E3 CADETS_E3)
TARGETS=(CADETS_E3 TRACE_E3)
SCENARIOS=(S1 S2)

RESULTS=/root/PIDSMaker/artifacts/transfer_results/all_results.csv
TASK_LOG_DIR=/root/PIDSMaker/artifacts/task_logs
mkdir -p /root/PIDSMaker/artifacts/transfer_results "$TASK_LOG_DIR"

if [ ! -f "$RESULTS" ]; then
    echo "system,scenario,source,target,TP,TN,FP,FN,MCC,F1,ADP" > "$RESULTS"
fi

echo "================================================================"
echo "# RUN S1+S2  start: $(date)"
echo "# pid: $$  host: $(hostname)"
echo "# parallel scheduling: GPU0=[flash, then velox]  GPU1=[orthrus]"
echo "# gpu:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv 2>/dev/null
echo "================================================================"

# ----------------------------------------------------------------------
# Phase 0: wait for trace_e3.dump to be fully downloaded, then restore
# ----------------------------------------------------------------------
DUMP=/root/PIDSMaker/data/trace_e3.dump

echo
echo "===== PHASE 0: wait for $DUMP, then pg_restore ====="
while true; do
    if [ -f "$DUMP" ] && ! ls /root/PIDSMaker/data/trace_e3.dump*.part >/dev/null 2>&1; then
        echo "[$(date '+%H:%M:%S')] dump fully downloaded: $(ls -la $DUMP | awk '{print $5}') bytes"
        break
    fi
    sleep 60
done

ROWS=$(su - pgsql -c "PATH=/root/miniconda3/envs/pids/bin:\$PATH psql -h 127.0.0.1 -U pgsql -d trace_e3 -t -c \"select count(*) from event_table\" 2>/dev/null" | tr -d ' ' | head -1)
if [ -n "$ROWS" ] && [ "$ROWS" != "0" ]; then
    echo "[$(date '+%H:%M:%S')] trace_e3 already restored ($ROWS rows in event_table). skip."
else
    echo "[$(date '+%H:%M:%S')] starting pg_restore (parallel -j 8)"
    chmod a+r "$DUMP"
    su - pgsql -c "PATH=/root/miniconda3/envs/pids/bin:\$PATH \
        pg_restore -h 127.0.0.1 -p 5432 -U pgsql -d trace_e3 \
        --no-owner --no-privileges -j 8 $DUMP" 2>&1
    echo "[$(date '+%H:%M:%S')] pg_restore done"
    su - pgsql -c "PATH=/root/miniconda3/envs/pids/bin:\$PATH psql -h 127.0.0.1 -U pgsql -d trace_e3 -c '\\dt+'" 2>&1
fi

# ----------------------------------------------------------------------
# Phase 1: parallel training
#   GPU0: flash -> (when flash done) velox
#   GPU1: orthrus
#   Wait for ALL three before phase 2.
# ----------------------------------------------------------------------
echo
echo "===== PHASE 1: parallel training (GPU0: flash->velox, GPU1: orthrus) ====="

run_train() {
    local det="$1"
    local gpu="$2"
    local log="$TASK_LOG_DIR/train_${det}_TRACE_E3.log"
    echo "[TRAIN start $(date '+%H:%M:%S')] $det on TRACE_E3 (GPU $gpu) -> $log"
    CUDA_VISIBLE_DEVICES="$gpu" python pidsmaker/main.py "$det" TRACE_E3 "${DB[@]}" "${ART[@]}" > "$log" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        echo "[TRAIN done  $(date '+%H:%M:%S')] $det on TRACE_E3 OK"
    else
        echo "[TRAIN FAIL  $(date '+%H:%M:%S')] $det on TRACE_E3 rc=$rc -- see $log"
        tail -8 "$log" | sed 's/^/    /'
    fi
    return $rc
}

# GPU0 chain: flash then velox
(
    run_train flash 0
    run_train velox 0
) &
PID_GPU0=$!

# GPU1: orthrus
(
    run_train orthrus 1
) &
PID_GPU1=$!

echo "GPU0 chain pid: $PID_GPU0   GPU1 (orthrus) pid: $PID_GPU1"

# Wait both
wait $PID_GPU0
wait $PID_GPU1
echo "===== PHASE 1 ALL DONE  $(date '+%H:%M:%S') ====="

# ----------------------------------------------------------------------
# Phase 2: 6 transfers in parallel (3 on GPU0, 3 on GPU1)
# Distribute by detector: GPU0={flash,velox}  GPU1={orthrus}
# Inside each GPU, run S1 then S2 (sequential, since each is fast).
# ----------------------------------------------------------------------
echo
echo "===== PHASE 2: parallel transfer ====="

run_transfer() {
    local det="$1"
    local gpu="$2"
    for i in "${!SOURCES[@]}"; do
        local src="${SOURCES[$i]}"
        local tgt="${TARGETS[$i]}"
        local sc="${SCENARIOS[$i]}"
        local log="$TASK_LOG_DIR/transfer_${det}_${sc}_${src}_to_${tgt}.log"

        echo "[TRANSFER start $(date '+%H:%M:%S')] [$sc] $det $src->$tgt (GPU $gpu)"
        CUDA_VISIBLE_DEVICES="$gpu" python transfer_inference.py "$det" "$src" "$tgt" "${ART[@]}" "${DB[@]}" > "$log" 2>&1
        local rc=$?
        local output
        output=$(cat "$log")

        local tp tn fp fn mcc f1 adp
        tp=$(echo  "$output" | grep "^.*- tp:"        | tail -1 | awk '{print $NF}')
        tn=$(echo  "$output" | grep "^.*- tn:"        | tail -1 | awk '{print $NF}')
        fp=$(echo  "$output" | grep "^.*- fp:"        | tail -1 | awk '{print $NF}')
        fn=$(echo  "$output" | grep "^.*- fn:"        | tail -1 | awk '{print $NF}')
        mcc=$(echo "$output" | grep "^.*- mcc:"       | tail -1 | awk '{print $NF}')
        f1=$(echo  "$output" | grep "^.*- fscore:"    | tail -1 | awk '{print $NF}')
        adp=$(echo "$output" | grep "^.*- adp_score:" | tail -1 | awk '{print $NF}')

        if [ -n "$tp" ]; then
            echo "[TRANSFER done  $(date '+%H:%M:%S')] [$sc] $det $src->$tgt OK TP=$tp FN=$fn MCC=$mcc F1=$f1 ADP=$adp"
        else
            local err
            err=$(echo "$output" | grep -E "Error|Killed|Traceback" | tail -1 | cut -c1-120)
            echo "[TRANSFER FAIL  $(date '+%H:%M:%S')] [$sc] $det $src->$tgt rc=$rc -- $err"
        fi
        echo "$det,$sc,$src,$tgt,$tp,$tn,$fp,$fn,$mcc,$f1,$adp" >> "$RESULTS"
    done
}

# GPU0 runs flash and velox transfers (sequentially, 4 transfers)
(
    run_transfer flash 0
    run_transfer velox 0
) &
T_GPU0=$!

# GPU1 runs orthrus transfers (2 transfers)
(
    run_transfer orthrus 1
) &
T_GPU1=$!

wait $T_GPU0
wait $T_GPU1

echo
echo "================================================================"
echo "# RUN S1+S2  done: $(date)"
echo "# results CSV (S1+S2 rows):"
grep -E "^[a-z]+,(S1|S2)," "$RESULTS"
echo "================================================================"
