#!/bin/bash
# PUDA Cross-Domain Transfer: All Detectors × All Scenarios
# Usage: bash run_all_transfer_experiments.sh [--db-password PASSWORD]

set -e
source /opt/conda/etc/profile.d/conda.sh && conda activate pids
cd /home/pids

DB_PASS="${1:-postgres}"
DB="--database_host postgres --database_port 5432 --database_user postgres --database_password $DB_PASS"
ART="--artifact_dir /home/artifacts"

DETECTORS="magic kairos flash orthrus velox"
DATASETS="CADETS_E3 THEIA_E3 optc_h501 optc_h201 optc_h051"

SOURCES=(CADETS_E3 THEIA_E3 THEIA_E3 optc_h501 optc_h201 CADETS_E3)
TARGETS=(THEIA_E3 CADETS_E3 optc_h501 THEIA_E3 CADETS_E3 optc_h051)
SCENARIOS=(S3 S4 S5 S6 S7 S8)

RESULTS=/home/artifacts/transfer_results/all_results.csv
mkdir -p /home/artifacts/transfer_results
echo "system,scenario,source,target,TP,TN,FP,FN,MCC,F1,ADP" > $RESULTS

for det in $DETECTORS; do
    echo ""
    echo "################################################################"
    echo "# DETECTOR: $det"
    echo "################################################################"
    
    # Phase 1: Train on all datasets (serially to avoid OOM)
    for ds in $DATASETS; do
        echo "[TRAIN] $det on $ds"
        python pidsmaker/main.py $det $ds $DB $ART 2>&1 | tail -1
        echo "[DONE] $det on $ds"
    done
    
    # Phase 2: Transfer experiments
    for i in "${!SOURCES[@]}"; do
        src="${SOURCES[$i]}"
        tgt="${TARGETS[$i]}"
        sc="${SCENARIOS[$i]}"
        
        echo "[$sc] $det: $src -> $tgt"
        output=$(python transfer_inference.py $det "$src" "$tgt" $ART $DB 2>&1)
        
        tp=$(echo "$output" | grep "^.*- tp:" | tail -1 | awk '{print $NF}')
        tn=$(echo "$output" | grep "^.*- tn:" | tail -1 | awk '{print $NF}')
        fp=$(echo "$output" | grep "^.*- fp:" | tail -1 | awk '{print $NF}')
        fn=$(echo "$output" | grep "^.*- fn:" | tail -1 | awk '{print $NF}')
        mcc=$(echo "$output" | grep "^.*- mcc:" | tail -1 | awk '{print $NF}')
        f1=$(echo "$output" | grep "^.*- fscore:" | tail -1 | awk '{print $NF}')
        adp=$(echo "$output" | grep "^.*- adp_score:" | tail -1 | awk '{print $NF}')
        
        if [ -n "$tp" ]; then
            echo "  ✓ TP=$tp FN=$fn MCC=$mcc F1=$f1 ADP=$adp"
        else
            err=$(echo "$output" | grep -E "Error|Killed" | tail -1 | cut -c1-80)
            echo "  ✗ FAIL: $err"
        fi
        echo "$det,$sc,$src,$tgt,$tp,$tn,$fp,$fn,$mcc,$f1,$adp" >> $RESULTS
    done
done

echo ""
echo "################################################################"
echo "# ALL EXPERIMENTS COMPLETE"
echo "# Results: $RESULTS"
echo "################################################################"
cat $RESULTS
