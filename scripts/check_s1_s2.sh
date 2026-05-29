#!/bin/bash
# Quick status of S1/S2 run (Trace dump dl + pg_restore + train + transfer)

echo "===== S1+S2 STATUS  $(date) ====="

# 1. gdown
GD_PID=$(pgrep -f "gdown.*1xZNBbhWQO0xGVBsg6ujdPh9UMUXiUQQd" | head -1)
if [ -n "$GD_PID" ]; then
  echo "[GDOWN ] running  pid=$GD_PID  uptime=$(ps -p $GD_PID -o etime= | xargs)"
  tail -c 800 /tmp/gdown_trace.log 2>/dev/null | tr '\r' '\n' | grep -E "%" | tail -1
else
  if [ -f /root/PIDSMaker/data/trace_e3.dump ]; then
    sz=$(ls -la /root/PIDSMaker/data/trace_e3.dump | awk '{print $5}')
    echo "[GDOWN ] DONE     dump_size=$sz bytes"
  else
    echo "[GDOWN ] not running and dump not present"
  fi
fi

# 2. trace_e3 DB rows
ROWS=$(su - pgsql -c "PATH=/root/miniconda3/envs/pids/bin:\$PATH psql -h 127.0.0.1 -U pgsql -d trace_e3 -t -c \"select count(*) from event_table\" 2>/dev/null" | tr -d ' ' | head -1)
echo "[DB    ] trace_e3.event_table rows: ${ROWS:-(table not yet)}"

# 3. main run pid
if [ -f /root/PIDSMaker/artifacts/run_s1_s2.pid ]; then
  P=$(cat /root/PIDSMaker/artifacts/run_s1_s2.pid)
  if ps -p $P >/dev/null 2>&1; then
    echo "[MAIN  ] RUNNING  pid=$P  uptime=$(ps -p $P -o etime= | xargs)"
  else
    echo "[MAIN  ] STOPPED  (last pid=$P)"
  fi
fi

# 4. last log lines
echo
echo "----- run_s1_s2.log tail -----"
tail -25 /root/PIDSMaker/artifacts/run_s1_s2.log 2>/dev/null

# 5. detector training progress (file size grows)
echo
echo "----- training logs -----"
for det in flash orthrus velox; do
  f=/root/PIDSMaker/artifacts/task_logs/train_${det}_TRACE_E3.log
  if [ -f "$f" ]; then
    printf "%-8s %s\n" "$det" "$(ls -la $f | awk '{print $5" bytes "$6" "$7" "$8}')   last: $(tail -1 $f 2>/dev/null | cut -c1-100)"
  fi
done

# 6. transfer_results CSV (filter S1/S2)
echo
echo "----- transfer_results (S1+S2 rows) -----"
grep -E "^[a-z]+,(S1|S2)," /root/PIDSMaker/artifacts/transfer_results/all_results.csv 2>/dev/null

# 7. GPU
echo
echo "----- GPU -----"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv 2>/dev/null
