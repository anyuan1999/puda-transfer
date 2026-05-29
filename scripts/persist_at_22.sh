#!/bin/bash
# Triggered at 22:00 (1h before container deadline 23:00).
# Steps:
#   1) Final rsync of artifacts -> /jizhicfs/PIDSMaker_root/artifacts
#   2) Tar all critical code+config files
#   3) Sync code_files/ overrides
#   4) Copy results CSV out
#   5) Generate RESTORE_v2.md (continuation guide)
#   6) Push everything (scripts/results/RESTORE_v2.md/inventory) to git branch
#      restore-v2-trace-s1s2 on https://github.com/anyuan1999/puda-transfer
#   7) Continue rsync every 10 min until 22:55

set -u
PERSIST=/jizhicfs/PIDSMaker_root
TS=$(date +%Y%m%d_%H%M%S)
LOG=$PERSIST/logs/persist_at_22.log

mkdir -p "$PERSIST/logs"
exec >> "$LOG" 2>&1

echo
echo "================================================================"
echo "[persist_at_22] START $(date)"
echo "================================================================"

# ----------------------------------------------------------------------
# 1. Final rsync of artifacts (use --delete-after to mirror exactly)
# ----------------------------------------------------------------------
echo "[$(date '+%H:%M:%S')] rsync artifacts -> $PERSIST/artifacts"
rsync -a --delete-after /root/PIDSMaker/artifacts/ "$PERSIST/artifacts/"
echo "[$(date '+%H:%M:%S')] rsync done. Sizes:"
du -sh /root/PIDSMaker/artifacts "$PERSIST/artifacts"

# ----------------------------------------------------------------------
# 2. Tar code+config (excluding artifacts/ which is already rsynced, and dumps)
# ----------------------------------------------------------------------
TAR_FILE="$PERSIST/code_and_models_${TS}.tar.gz"
echo "[$(date '+%H:%M:%S')] tar -> $TAR_FILE"
cd /
tar --exclude='root/PIDSMaker/data/*.dump' \
    --exclude='root/PIDSMaker/.git' \
    --exclude='root/PIDSMaker/artifacts' \
    --exclude='root/PIDSMaker/__pycache__' \
    --exclude='root/PIDSMaker/**/__pycache__' \
    --exclude='root/puda-transfer*/.git' \
    -czf "$TAR_FILE" \
    root/PIDSMaker \
    root/puda-transfer \
    root/puda-transfer.gitclone 2>&1 | tail -3
ls -lh "$TAR_FILE"

# ----------------------------------------------------------------------
# 3. Sync code_files dir (overwrite-friendly snapshot of patches & scripts)
# ----------------------------------------------------------------------
echo "[$(date '+%H:%M:%S')] sync code_files/ overrides"
mkdir -p "$PERSIST/code_files"
cp -f /root/PIDSMaker/run_all_transfer_experiments.sh \
      /root/PIDSMaker/run_orthrus_only.sh \
      /root/PIDSMaker/run_velox_only.sh \
      /root/PIDSMaker/run_s1_s2.sh \
      /root/PIDSMaker/run_orthrus_s1_s2.sh \
      /root/PIDSMaker/run_flash_s1_s2.sh \
      /root/PIDSMaker/run_velox_after_flash.sh \
      /root/PIDSMaker/transfer_inference.py \
      /root/PIDSMaker/check_progress.sh \
      /root/PIDSMaker/check_s1_s2.sh \
      /root/PIDSMaker/scripts_persist/persist_at_22.sh \
      /root/PIDSMaker/pidsmaker/detection/training_methods/training_loop.py \
      /root/PIDSMaker/pidsmaker/utils/data_utils.py \
      /root/PIDSMaker/pidsmaker/featurization/feat_inference_methods/feat_inference_flash.py \
      /root/PIDSMaker/config/orthrus.yml \
      "$PERSIST/code_files/" 2>&1 | tail -5

# ----------------------------------------------------------------------
# 4. Final results CSV snapshot
# ----------------------------------------------------------------------
cp -f /root/PIDSMaker/artifacts/transfer_results/all_results.csv "$PERSIST/all_results_${TS}.csv" 2>/dev/null && \
    echo "[$(date '+%H:%M:%S')] results CSV copied: $PERSIST/all_results_${TS}.csv"

# ----------------------------------------------------------------------
# 5. Determine current state for RESTORE_v2.md
# ----------------------------------------------------------------------

# Helper: did detector X complete training on TRACE_E3?
detector_train_done() {
    local det=$1
    local f="/root/PIDSMaker/artifacts/task_logs/train_${det}_TRACE_E3.log"
    [ -f "$f" ] || { echo "0"; return; }
    if grep -q -E "Test Evaluation|Resutls saved|TRANSFER COMPLETE" "$f" 2>/dev/null; then
        echo "1"
    else
        echo "0"
    fi
}
# How many model_epoch_* exist for this detector on TRACE_E3?
detector_model_epoch_count() {
    local hash_dir
    hash_dir=$(grep -lE "main\.py.*$1.*TRACE_E3|$1.*TRACE_E3" /root/PIDSMaker/artifacts/training/training/*/TRACE_E3/logs/*.log 2>/dev/null | head -1 | xargs dirname 2>/dev/null)
    # Fallback: just list any TRACE_E3 hash dir's gnn_models
    find /root/PIDSMaker/artifacts/training -path "*TRACE_E3/gnn_models/model_epoch_*" -type d 2>/dev/null | wc -l
}

ORTHRUS_DONE=$(detector_train_done orthrus)
FLASH_DONE=$(detector_train_done flash)
VELOX_DONE=$(detector_train_done velox)
S1S2_ROWS=$(grep -E "^[a-z]+,(S1|S2)," /root/PIDSMaker/artifacts/transfer_results/all_results.csv 2>/dev/null | wc -l)

# Inventory of all model_epoch dirs for TRACE_E3
INVENTORY="$PERSIST/artifacts_inventory_${TS}.txt"
{
    echo "# Artifacts inventory for TRACE_E3 (timestamp: ${TS})"
    echo
    echo "## TRACE_E3 model_epoch_* paths (relative to /root/PIDSMaker/artifacts/training/training/)"
    find /root/PIDSMaker/artifacts/training -path "*TRACE_E3/gnn_models/model_epoch_*" -type d 2>/dev/null \
        | sed 's|/root/PIDSMaker/artifacts/training/training/||' | sort
    echo
    echo "## TRACE_E3 stage outputs"
    for stage in construction transformation featurization feat_inference batching; do
        echo "### $stage"
        find /root/PIDSMaker/artifacts/$stage -path "*TRACE_E3*" -type d 2>/dev/null \
            | sed "s|/root/PIDSMaker/artifacts/$stage/||" | sort | head -10
    done
} > "$INVENTORY"
ls -la "$INVENTORY"

# ----------------------------------------------------------------------
# 6. Generate RESTORE_v2.md
# ----------------------------------------------------------------------
RESTORE_V2="$PERSIST/RESTORE_v2.md"
cat > "$RESTORE_V2" <<MARKDOWN
# PIDSMaker / PUDA Transfer — Restore Guide v2 (Continuation)

> 用于在新机器上**继续 S1/S2 实验**（接着上一个容器的状态）。
> 上一个容器到期：2026-05-29 23:00。本快照 timestamp: \`${TS}\`
>
> **精简版 quickstart 在 §2，详细机理在 §0-§1**

## 0. 这次新增的内容

* 新数据集：**TRACE_E3**（DARPA E3 Trace, Linux）
  * gdrive id: \`1xZNBbhWQO0xGVBsg6ujdPh9UMUXiUQQd\`（11.5 GB dump）
  * postgres db 名: \`trace_e3\`
* 新 transfer scenarios：
  * **S1**: \`TRACE_E3\` (Linux)    -> \`CADETS_E3\` (FreeBSD)
  * **S2**: \`CADETS_E3\` (FreeBSD) -> \`TRACE_E3\`  (Linux)
* 新脚本（已 push 到 git: \`restore-v2-trace-s1s2\` 分支 + 在 \`code_files/\`）：
  * \`run_orthrus_s1_s2.sh\` — 单 GPU (1) 跑 orthrus × {S1,S2}
  * \`run_flash_s1_s2.sh\`   — 单 GPU (0) 跑 flash × {S1,S2}
  * \`run_velox_after_flash.sh\` — watcher，等 flash 退出后接管 GPU0 跑 velox
  * \`run_s1_s2.sh\`          — 双 GPU 一起调度版（备用）
  * \`check_s1_s2.sh\`        — 进度查看
  * \`scripts_persist/persist_at_22.sh\` — 22:00 自动持久化脚本

## 1. 上个容器到期前的状态

| 项目 | 状态 |
|---|---|
| TRACE_E3 dump 已下载 | 🚫 新机器需要重下（\`/root/PIDSMaker/data/trace_e3.dump\` 不在 jizhicfs）|
| trace_e3 PG 数据库 | 🚫 新机器需要 createdb + pg_restore |
| **TRAIN orthrus on TRACE_E3** | $([ "$ORTHRUS_DONE" = "1" ] && echo "✅ 已完成（命中即跳）" || echo "⚠️ 未完成（需续训或重训）") |
| **TRAIN flash on TRACE_E3**   | $([ "$FLASH_DONE" = "1" ] && echo "✅ 已完成（命中即跳）" || echo "⚠️ 未完成（需续训或重训）") |
| **TRAIN velox on TRACE_E3**   | $([ "$VELOX_DONE" = "1" ] && echo "✅ 已完成（命中即跳）" || echo "⚠️ 未完成（需续训或重训）") |
| transfer_results 中 S1/S2 行数 | **${S1S2_ROWS}** 行 |

详细 model_epoch 清单见 \`artifacts_inventory_${TS}.txt\`

## 2. Quickstart：在新机器上 5 步搞定

> **前提：你已经按原始 \`RESTORE.md\` §1.1-§1.5 完成基础环境**：conda env、torch、
> pyg、postgresql=17、pgsql 用户、5 个数据库（cadets_e3, theia_e3, optc_051,
> optc_201, optc_501）已经齐全。如果你是全新机器，**先跑原 RESTORE.md 的全部
> 步骤**，再回到这里。

### 2.1 拉取最新代码（含本次所有脚本）

\`\`\`bash
cd /root
# 如果 puda-transfer 不存在
git clone https://github.com/anyuan1999/puda-transfer.git
cd puda-transfer
git checkout restore-v2-trace-s1s2   # ← 本次新分支
cat RESTORE_v2.md                    # ← 这份文档
\`\`\`

### 2.2 把脚本和补丁覆盖到 PIDSMaker

\`\`\`bash
SRC=/root/puda-transfer
DST=/root/PIDSMaker

# patches (4 个补丁文件)
cp \$SRC/patches/training_loop.py        \$DST/pidsmaker/detection/training_methods/training_loop.py
cp \$SRC/patches/data_utils.py           \$DST/pidsmaker/utils/data_utils.py
cp \$SRC/patches/feat_inference_flash.py \$DST/pidsmaker/featurization/feat_inference_methods/feat_inference_flash.py
cp \$SRC/patches/orthrus.yml             \$DST/config/orthrus.yml

# scripts (拷到 PIDSMaker 根)
cp \$SRC/scripts/*.sh \$DST/
chmod +x \$DST/*.sh

# editable reinstall
cd \$DST && pip install -e .
\`\`\`

### 2.3 创建 trace_e3 DB + 重下 dump + restore

\`\`\`bash
su - pgsql -c "PATH=/root/miniconda3/envs/pids/bin:\\\$PATH createdb -h 127.0.0.1 -U pgsql trace_e3"

source /root/miniconda3/etc/profile.d/conda.sh && conda activate pids
cd /root/PIDSMaker && mkdir -p data
gdown 1xZNBbhWQO0xGVBsg6ujdPh9UMUXiUQQd -O data/trace_e3.dump --fuzzy   # ~20 min @ 10MB/s

chmod a+r data/trace_e3.dump
su - pgsql -c "PATH=/root/miniconda3/envs/pids/bin:\\\$PATH \\
    pg_restore -h 127.0.0.1 -p 5432 -U pgsql -d trace_e3 \\
    --no-owner --no-privileges -j 8 /root/PIDSMaker/data/trace_e3.dump"
\`\`\`

### 2.4 恢复 artifacts（**关键节省时间**点）

\`\`\`bash
# rsync 上一容器训练好的所有 artifacts 到本地
rsync -a /jizhicfs/PIDSMaker_root/artifacts/ /root/PIDSMaker/artifacts/
# ~250-280 GB，约 30 min
\`\`\`

完成后 PIDSMaker 会通过 cfg-hash 自动命中已经训过的 model：
$([ "$ORTHRUS_DONE" = "1" ] && echo "* orthrus on TRACE_E3 ✅ 直接命中，无需重训")
$([ "$FLASH_DONE" = "1" ] && echo "* flash on TRACE_E3 ✅ 直接命中，无需重训")
$([ "$VELOX_DONE" = "1" ] && echo "* velox on TRACE_E3 ✅ 直接命中，无需重训")

### 2.5 续跑

\`\`\`bash
cd /root/PIDSMaker
source /root/miniconda3/etc/profile.d/conda.sh && conda activate pids
export OMP_NUM_THREADS=8

# 看进度
bash check_s1_s2.sh

# 如果某个 detector 没训完，对应启动（会接着 cfg-hash artifacts 跑，缺什么补什么）
$([ "$ORTHRUS_DONE" != "1" ] && echo "nohup bash run_orthrus_s1_s2.sh > artifacts/run_orthrus_s1_s2.log 2>&1 &")
$([ "$FLASH_DONE" != "1" ] && echo "nohup bash run_flash_s1_s2.sh > artifacts/run_flash_s1_s2.log 2>&1 &")
$([ "$VELOX_DONE" != "1" ] && echo "# velox 通常等 flash 完成后再启动:\nnohup bash run_velox_after_flash.sh > artifacts/run_velox_after_flash.log 2>&1 &")

# 重启 sync daemon (持续 5 min 同步)
nohup bash scripts_persist/sync_to_persist.sh > /jizhicfs/PIDSMaker_root/logs/sync_daemon.log 2>&1 &

# 视新容器到期时间，调度新一轮 22:00 持久化（修改脚本里的时间，或自定 sleep）
\`\`\`

## 3. 计时数据（实测）

| 阶段 | 耗时 |
|---|---|
| gdown trace_e3.dump | ~20 min |
| pg_restore -j 8 | ~10 min |
| rsync artifacts (~280 GB) | ~30 min |
| TRAIN orthrus on TRACE_E3 (单 H20, build_graphs+feat_inference+training+eval) | $([ "$ORTHRUS_DONE" = "1" ] && echo "~3h（实测，14:06 → ~17:00）" || echo "估 3-4h") |
| TRAIN flash on TRACE_E3 | 估 3-5h |
| TRAIN velox on TRACE_E3 | 估 3-5h |
| transfer S1 / S2 (per detector) | 1-3 min |

## 4. 文件位置（速查）

| 项目 | 路径 |
|---|---|
| 持久备份根 | \`/jizhicfs/PIDSMaker_root/\` |
| 本次 tar | \`code_and_models_${TS}.tar.gz\` |
| 历史 tar | \`code_and_models_20260528_164928.tar.gz\` (5/28 原始) |
| 结果 CSV (本次) | \`all_results_${TS}.csv\` |
| 最新结果 CSV (滚动) | \`all_results_latest.csv\` |
| artifacts inventory | \`artifacts_inventory_${TS}.txt\` |
| artifacts 主体 | \`artifacts/\` |
| 代码补丁 | \`code_files/\` |
| 旧 RESTORE | \`RESTORE.md\` (端到端从 0 恢复) |
| 新 RESTORE | \`RESTORE_v2.md\` (本文件，S1/S2 增量恢复) |
| Git 仓库 | \`https://github.com/anyuan1999/puda-transfer\`（分支 \`restore-v2-trace-s1s2\`） |

## 5. 申请新机器的 Checklist

- [ ] **资源池**：必须与 jizhicfs 同池（否则看不到 \`/jizhicfs/PIDSMaker_root\`）
- [ ] **GPU**：≥ 1 张 H20/H100/A100，建议 2 张
- [ ] **内存**：≥ 256 GB
- [ ] **磁盘**：root 至少 500 GB 空闲（artifacts 约 280 GB）
- [ ] **挂载验证**：\`ls /jizhicfs/PIDSMaker_root/\` 应能看到 \`artifacts/\` \`RESTORE_v2.md\` 等
- [ ] **GLIBC**：< 2.29 没事，PyG 2.6 自动 fallback；warnings 可忽略

## 6. 如果 jizhicfs 也丢了（最坏情况）

最坏情况下持久盘也访问不了，从 git 仓库的 \`restore-v2-trace-s1s2\` 分支可以恢复：
1. 仓库里有所有的代码补丁（\`patches/\`）+ 所有脚本（\`scripts/\`）+ \`RESTORE_v2.md\`
2. 但**模型权重必须重训**（artifacts 太大，无法存 git）
3. 重新跑 \`run_orthrus_s1_s2.sh\` 等，从头训
MARKDOWN

ls -la "$RESTORE_V2"

# ----------------------------------------------------------------------
# 7. Git push (key new step) — push to restore-v2-trace-s1s2 branch
# ----------------------------------------------------------------------
echo
echo "[$(date '+%H:%M:%S')] preparing git push to restore-v2-trace-s1s2 branch"

GIT_WORK=/root/puda-transfer.gitclone
cd "$GIT_WORK"

# Pull latest first (in case anything was pushed elsewhere)
git fetch origin 2>&1 | tail -3
# Create branch fresh from current restore-guide
git checkout -B restore-v2-trace-s1s2 origin/restore-guide 2>&1 | tail -3

# Sync new files in
mkdir -p patches scripts
cp -f /root/PIDSMaker/pidsmaker/detection/training_methods/training_loop.py     patches/
cp -f /root/PIDSMaker/pidsmaker/utils/data_utils.py                              patches/
cp -f /root/PIDSMaker/pidsmaker/featurization/feat_inference_methods/feat_inference_flash.py patches/
cp -f /root/PIDSMaker/config/orthrus.yml                                         patches/

cp -f /root/PIDSMaker/run_all_transfer_experiments.sh   scripts/
cp -f /root/PIDSMaker/run_orthrus_only.sh               scripts/
cp -f /root/PIDSMaker/run_velox_only.sh                 scripts/
cp -f /root/PIDSMaker/run_s1_s2.sh                      scripts/
cp -f /root/PIDSMaker/run_orthrus_s1_s2.sh              scripts/
cp -f /root/PIDSMaker/run_flash_s1_s2.sh                scripts/
cp -f /root/PIDSMaker/run_velox_after_flash.sh          scripts/
cp -f /root/PIDSMaker/check_progress.sh                 scripts/
cp -f /root/PIDSMaker/check_s1_s2.sh                    scripts/
cp -f /root/PIDSMaker/scripts_persist/persist_at_22.sh  scripts/
cp -f /root/PIDSMaker/scripts_persist/sync_to_persist.sh scripts/
cp -f /root/PIDSMaker/scripts_persist/final_snapshot.sh  scripts/
cp -f /root/PIDSMaker/transfer_inference.py             ./

# Results + RESTORE_v2.md
mkdir -p results
cp -f /root/PIDSMaker/artifacts/transfer_results/all_results.csv  "results/all_results_${TS}.csv"
cp -f "$RESTORE_V2"  ./RESTORE_v2.md
cp -f "$INVENTORY"   "results/artifacts_inventory_${TS}.txt"

# Stage + commit
git add -A
git status --short 2>&1 | head -30
git commit -m "Persist S1/S2 progress at ${TS}

- Add run_orthrus_s1_s2.sh / run_flash_s1_s2.sh / run_velox_after_flash.sh
- Add check_s1_s2.sh + scripts_persist/persist_at_22.sh
- Add results snapshot all_results_${TS}.csv
- Add RESTORE_v2.md continuation guide

Train completion at persist time:
  orthrus on TRACE_E3: $([ "$ORTHRUS_DONE" = "1" ] && echo done || echo in-progress)
  flash on TRACE_E3:   $([ "$FLASH_DONE" = "1" ] && echo done || echo in-progress)
  velox on TRACE_E3:   $([ "$VELOX_DONE" = "1" ] && echo done || echo in-progress)
S1/S2 transfer rows in CSV: ${S1S2_ROWS}" 2>&1 | tail -10

# Push (force-with-lease for safety; this branch is dedicated)
git push -u origin restore-v2-trace-s1s2 --force-with-lease 2>&1 | tail -10
echo "[$(date '+%H:%M:%S')] git push done"

# Drop back to where we were (don't leave dirty checkout)
git checkout restore-guide 2>&1 | tail -3

# ----------------------------------------------------------------------
# 8. Trigger more rsync passes every 10 min until 22:55 + final push at 22:50
# ----------------------------------------------------------------------
echo "[$(date '+%H:%M:%S')] schedule extra rsync passes every 10min until 22:55"
END_TS=$(date -d '2026-05-29 22:55:00' +%s)
PUSH_AGAIN_TS=$(date -d '2026-05-29 22:50:00' +%s)
PUSHED_AGAIN=0
while [ "$(date +%s)" -lt "$END_TS" ]; do
    sleep 600
    echo "[$(date '+%H:%M:%S')] rsync pass"
    rsync -a --delete-after /root/PIDSMaker/artifacts/ "$PERSIST/artifacts/" 2>&1 | tail -3
    cp -f /root/PIDSMaker/artifacts/transfer_results/all_results.csv "$PERSIST/all_results_latest.csv" 2>/dev/null

    # 22:50 second git push with latest CSV
    if [ "$PUSHED_AGAIN" -eq 0 ] && [ "$(date +%s)" -ge "$PUSH_AGAIN_TS" ]; then
        echo "[$(date '+%H:%M:%S')] second git push with final CSV"
        cd "$GIT_WORK"
        git checkout restore-v2-trace-s1s2 2>&1 | tail -2
        TS2=$(date +%Y%m%d_%H%M%S)
        cp -f /root/PIDSMaker/artifacts/transfer_results/all_results.csv  "results/all_results_${TS2}_FINAL.csv"
        git add -A
        git commit -m "Final results snapshot at ${TS2}" 2>&1 | tail -5
        git push origin restore-v2-trace-s1s2 2>&1 | tail -5
        git checkout restore-guide 2>&1 | tail -1
        PUSHED_AGAIN=1
    fi
done

echo
echo "[$(date '+%H:%M:%S')] FINAL rsync (last pass before deadline)"
rsync -a --delete-after /root/PIDSMaker/artifacts/ "$PERSIST/artifacts/" 2>&1 | tail -3
cp -f /root/PIDSMaker/artifacts/transfer_results/all_results.csv "$PERSIST/all_results_FINAL.csv" 2>/dev/null
echo "[$(date '+%H:%M:%S')] persist_at_22 DONE"
echo "================================================================"
