#!/bin/bash
# 查看实验进度: bash /root/PIDSMaker/check_progress.sh
echo "===== RUN_ALL STATUS $(date) ====="
if [ -f /root/PIDSMaker/artifacts/run_all.pid ]; then
  P=$(cat /root/PIDSMaker/artifacts/run_all.pid)
  if ps -p $P > /dev/null 2>&1; then
    echo "STATUS: RUNNING (PID=$P, uptime=$(ps -p $P -o etime= | xargs))"
  else
    echo "STATUS: STOPPED (PID=$P not running)"
  fi
fi
echo ""
echo "===== Current task ====="
ps -ef | grep -E "pidsmaker/main|transfer_inference" | grep -v grep | head -3
echo ""
echo "===== Last 20 log lines ====="
tail -20 /root/PIDSMaker/artifacts/run_all.log
echo ""
echo "===== Saved models (per detector × dataset) ====="
find /root/PIDSMaker/artifacts/training -name "model_epoch_*" -type d 2>/dev/null | sort | sed 's|/root/PIDSMaker/artifacts/training/training/||'
echo ""
echo "===== Transfer results (if any) ====="
if [ -f /root/PIDSMaker/artifacts/transfer_results/all_results.csv ]; then
  cat /root/PIDSMaker/artifacts/transfer_results/all_results.csv
fi
echo ""
echo "===== GPU ====="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv 2>/dev/null | head -5
