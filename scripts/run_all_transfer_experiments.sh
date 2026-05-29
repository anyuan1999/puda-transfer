#!/bin/bash
# PUDA Cross-Domain Transfer: All Detectors × All Scenarios
# Usage: bash run_all_transfer_experiments.sh [--db-password PASSWORD]

# NOTE: no `set -e` -- we want to keep going if one detector/dataset fails.
# (conda already activated)
cd /root/PIDSMaker

DB_PASS="${1:-}"
DB=(--database_host 127.0.0.1 --database_port 5432 --database_user pgsql --database_password "")
ART=(--artifact_dir /root/PIDSMaker/artifacts)

DETECTORS="flash orthrus velox"
DATASETS="CADETS_E3 THEIA_E3 optc_h501 optc_h201 optc_h051"

SOURCES=(CADETS_E3 THEIA_E3 THEIA_E3 optc_h501 optc_h201 CADETS_E3)
TARGETS=(THEIA_E3 CADETS_E3 optc_h501 THEIA_E3 CADETS_E3 optc_h051)
SCENARIOS=(S3 S4 S5 S6 S7 S8)

RESULTS=/root/PIDSMaker/artifacts/transfer_results/all_results.csv
TASK_LOG_DIR=/root/PIDSMaker/artifacts/task_logs
mkdir -p /root/PIDSMaker/artifacts/transfer_results "$TASK_LOG_DIR"

# Only write CSV header if file doesn't exist (allow resume / merge)
if [ ! -f "$RESULTS" ]; then
    echo "system,scenario,source,target,TP,TN,FP,FN,MCC,F1,ADP" > "$RESULTS"
fi

for det in $DETECTORS; do
    echo ""
    echo "################################################################"
    echo "# DETECTOR: $det  ($(date))"
    echo "################################################################"

    # Phase 1: Train on all datasets (serially to avoid OOM)
    for ds in $DATASETS; do
        log="$TASK_LOG_DIR/train_${det}_${ds}.log"
        echo "[TRAIN] $det on $ds  -> $log  ($(date +%H:%M:%S))"
        python pidsmaker/main.py $det $ds "${DB[@]}" "${ART[@]}" > "$log" 2>&1
        rc=$?
        if [ $rc -eq 0 ]; then
            echo "[DONE]  $det on $ds  ($(date +%H:%M:%S))"
        else
            echo "[FAIL]  $det on $ds rc=$rc  -- see $log"
            tail -5 "$log" | sed 's/^/    /'
        fi
    done

    # Phase 2: Transfer experiments
    for i in "${!SOURCES[@]}"; do
        src="${SOURCES[$i]}"
        tgt="${TARGETS[$i]}"
        sc="${SCENARIOS[$i]}"
        log="$TASK_LOG_DIR/transfer_${det}_${sc}_${src}_to_${tgt}.log"

        echo "[$sc] $det: $src -> $tgt  -> $log  ($(date +%H:%M:%S))"
        python transfer_inference.py $det "$src" "$tgt" "${ART[@]}" "${DB[@]}" > "$log" 2>&1
        rc=$?
        output=$(cat "$log")

        tp=$(echo "$output" | grep "^.*- tp:" | tail -1 | awk '{print $NF}')
        tn=$(echo "$output" | grep "^.*- tn:" | tail -1 | awk '{print $NF}')
        fp=$(echo "$output" | grep "^.*- fp:" | tail -1 | awk '{print $NF}')
        fn=$(echo "$output" | grep "^.*- fn:" | tail -1 | awk '{print $NF}')
        mcc=$(echo "$output" | grep "^.*- mcc:" | tail -1 | awk '{print $NF}')
        f1=$(echo "$output" | grep "^.*- fscore:" | tail -1 | awk '{print $NF}')
        adp=$(echo "$output" | grep "^.*- adp_score:" | tail -1 | awk '{print $NF}')

        if [ -n "$tp" ]; then
            echo "  OK TP=$tp FN=$fn MCC=$mcc F1=$f1 ADP=$adp"
        else
            err=$(echo "$output" | grep -E "Error|Killed|Traceback" | tail -1 | cut -c1-100)
            echo "  FAIL rc=$rc -- $err"
        fi
        echo "$det,$sc,$src,$tgt,$tp,$tn,$fp,$fn,$mcc,$f1,$adp" >> "$RESULTS"
    done
done

echo ""
echo "################################################################"
echo "# ALL EXPERIMENTS COMPLETE  ($(date))"
echo "# Results: $RESULTS"
echo "################################################################"
cat "$RESULTS"
