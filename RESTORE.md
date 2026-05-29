# PIDSMaker / PUDA Transfer — Restore Guide

> 用于在新机器（K8s 容器到期后重新申请的资源）上**恢复 PUDA 跨域迁移实验**。
> 本快照位置：`/jizhicfs/PIDSMaker_root/`（腾讯智算 CFS 共享盘）。
> 创建时间：2026-05-28 17:00（一次完整 rsync）+ 持续 5 分钟增量同步。

---

## 0. 这份快照里有什么

```
/jizhicfs/PIDSMaker_root/
├── artifacts/                              # 训练 artifacts（含 gnn_models, edge_losses, transfer_results 等）
│   ├── construction/<DATASET>/...          # build_graphs 中间产物（PostgreSQL → networkx 图）
│   ├── transformation/<DATASET>/...
│   ├── featurization/<DATASET>/...         # word2vec 模型
│   ├── feat_inference/<DATASET>/...        # 节点 embedding
│   ├── batching/...
│   ├── training/training/<HASH>/<DATASET>/
│   │   ├── gnn_models/model_epoch_{0..N}/  # ← 我们的补丁产出，这是迁移核心
│   │   └── edge_losses/{val,test}/model_epoch_{...}/
│   ├── transfer_results/all_results.csv    # ← Phase 2 最终指标
│   ├── task_logs/                          # 每个 detector × dataset 的详细日志
│   ├── run_all.log / run_orthrus.log / ... # 主控日志
├── code_files/                             # 关键已修改文件（覆盖回 PIDSMaker 源码即可）
│   ├── training_loop.py                    # 启用 model 保存（PUDA 补丁）
│   ├── data_utils.py                       # save_model TGN 容错
│   ├── feat_inference_flash.py             # PositionalEncoder 优化（避免每个 node 重复 init）
│   ├── orthrus.yml                         # word2vec.num_workers: 1 → 32
│   ├── run_all_transfer_experiments.sh     # 跨域迁移主脚本（已适配 127.0.0.1 + pgsql + 本地 artifacts）
│   ├── run_orthrus_only.sh                 # orthrus 独立并行脚本
│   ├── transfer_inference.py               # 跨域 inference 入口
│   ├── check_progress.sh                   # 一键查看进度
│   └── (sync/snapshot/watcher 三件套脚本)
├── code_and_models_20260528_164928.tar.gz  # 完整代码 tar 备份（包括 puda-transfer）
├── full_pidsmaker_<TS>.tar.gz              # final_snapshot 触发后会生成（可能没有）
├── all_results_<TS>.csv                    # final_snapshot 时 transfer 结果的副本
└── logs/sync.log, deadline_watcher.log     # daemon 日志
```

---

## 1. 在新机器上恢复（端到端）

### 1.1 申请新机器，确认 `/jizhicfs` 已挂载

新容器开起来后第一件事：

```bash
ls /jizhicfs/PIDSMaker_root/    # 应看到 artifacts/ code_files/ ...
```

如果**没有**该目录，说明：
- (a) 新机器不在同一资源池 → 联系运维换池
- (b) 挂载点变了 → `df -hT | grep -i fuse` 找新路径

> 备选持久化盘：`/apdcephfs_hzlf/share_1227201`（21 TB Ceph）也曾可用。

### 1.2 检查 GPU、内存、磁盘

```bash
nvidia-smi | head -20                  # 期望 ≥1 张 H20/H100/A100
free -h | head -2                      # 期望 ≥256 GB
df -h /                                # 容器根盘 ≥500 GB
```

### 1.3 安装 conda + 创建 pids env（约 15 分钟）

如果新容器自带 miniconda3 且 PATH 里有 conda，就跳过第 1 步。

```bash
# 1) 装 miniconda（如果没有）
if ! command -v conda &>/dev/null; then
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/m.sh
    bash /tmp/m.sh -b -p $HOME/miniconda3
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    conda init bash && source ~/.bashrc
fi

# 2) 创建 env + 装依赖
conda create -n pids python=3.10 -y
conda activate pids
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu124
pip install torch_geometric==2.6.0 --no-cache-dir
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.4.0+cu124.html --no-cache-dir
conda install -y psycopg2
conda install -y -c conda-forge "postgresql=17"      # 必须 17，因为 dump 是 v1.16 格式
pip install scikit-learn==1.2.0 networkx==2.8.7 xxhash==3.2.0 \
    graphviz==0.20.1 psutil scipy==1.10.1 matplotlib==3.8.4 \
    wandb==0.24.1 chardet==5.2.0 nltk==3.8.1 igraph==0.11.5 \
    cairocffi==1.7.0 wget==3.2 gensim==4.3.1 pytz==2024.1 \
    pandas==2.2.2 yacs==0.1.8 gdown==5.2.0 py-spy
pip install numpy==1.26.4 scipy==1.10.1
```

> 如果 `pyg_lib`/`torch_sparse` 因 glibc < 2.29 加载失败（TencentOS 3.2 系），**忽略**——PyG 2.6.0 自动 fallback。我们之前实验跑通了。

验证：

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# → True NVIDIA H20
```

### 1.4 恢复 PIDSMaker 代码 + 我们的补丁

最快方式：解压 tar 备份。

```bash
mkdir -p /root && cd /root
tar -xzf /jizhicfs/PIDSMaker_root/code_and_models_20260528_164928.tar.gz
# 这会创建 /root/PIDSMaker 和 /root/puda-transfer

cd /root/PIDSMaker
pip install -e .                          # 必须重做，因为是 site-packages 引用
```

或者从 git clone 重建（仅当 tar 不可用）：

```bash
cd /root
git clone https://github.com/ubc-provenance/PIDSMaker.git
cd PIDSMaker && git checkout tags/2.1.0 -b v2.1.0
git clone https://github.com/anyuan1999/puda-transfer.git ../puda-transfer
cp ../puda-transfer/transfer_inference.py .
cp ../puda-transfer/run_all_transfer_experiments.sh .

# 把 code_files 里所有补丁覆盖回去
cp /jizhicfs/PIDSMaker_root/code_files/training_loop.py            pidsmaker/detection/training_methods/
cp /jizhicfs/PIDSMaker_root/code_files/data_utils.py               pidsmaker/utils/
cp /jizhicfs/PIDSMaker_root/code_files/feat_inference_flash.py     pidsmaker/featurization/feat_inference_methods/
cp /jizhicfs/PIDSMaker_root/code_files/orthrus.yml                 config/
cp /jizhicfs/PIDSMaker_root/code_files/run_all_transfer_experiments.sh .
cp /jizhicfs/PIDSMaker_root/code_files/run_orthrus_only.sh         .   # 如果存在
cp /jizhicfs/PIDSMaker_root/code_files/run_velox_only.sh           .   # 如果存在
cp /jizhicfs/PIDSMaker_root/code_files/transfer_inference.py       .
cp /jizhicfs/PIDSMaker_root/code_files/check_progress.sh           .
chmod +x run_*.sh check_progress.sh
pip install -e .
```

### 1.5 恢复 PostgreSQL（约 15 分钟）

PostgreSQL 数据库**没有**在快照里（数据来自原始 dump，重建即可）。

```bash
# 1) 用非 root 用户跑 PG（initdb 拒 root）
useradd -m -s /bin/bash pgsql 2>/dev/null
chmod o+x /root /root/miniconda3 /root/miniconda3/envs /root/miniconda3/envs/pids /root/miniconda3/envs/pids/bin

# 2) initdb + 改配置 + 启动
su - pgsql -c '
set -e
export PATH=/root/miniconda3/envs/pids/bin:$PATH
export PGDATA=/home/pgsql/pgdata
rm -rf "$PGDATA"
initdb -D "$PGDATA" -U pgsql --auth-local=trust --auth-host=trust
echo "host all all 127.0.0.1/32 trust" >> "$PGDATA/pg_hba.conf"
cat >> "$PGDATA/postgresql.conf" <<EOF
listen_addresses = '"'"'127.0.0.1'"'"'
port = 5432
shared_buffers = 4GB
maintenance_work_mem = 2GB
EOF
pg_ctl -D "$PGDATA" -l /home/pgsql/pg.log start
sleep 4
for db in cadets_e3 theia_e3 optc_051 optc_201 optc_501; do createdb -h 127.0.0.1 -U pgsql "$db"; done
'
```

### 1.6 重新下载 + 加载 dump（约 20 分钟）

5 个 dump 共 7.7 GB，约 13 分钟下载（gdown ~10 MB/s）+ 4 分钟 restore。

```bash
cd /root/PIDSMaker
mkdir -p data
declare -A FILES=(
  [cadets_e3.dump]=1DGcGBhpavNmXTnCDd_s4NWBNh2n4-6nd
  [theia_e3.dump]=1p7HoH5SdMEFi0vkrEtMeG7B-hpw741p6
  [optc_h501.dump]=1046BVjpMql1bb5WHr9yQeB6Uq6RpngbM
  [optc_h201.dump]=1OSZXCQrocFSRN7wkPM02p-BqE2WmgdLD
  [optc_h051.dump]=1zzwge346AdAdxldykZOai5FtUrIaWc6q
)
for name in "${!FILES[@]}"; do
  gdown "${FILES[$name]}" -O "data/$name" --fuzzy
done

# Restore (注意 dump 文件名 -> 数据库名映射)
chmod a+r data/*.dump
declare -A MAP=([cadets_e3.dump]=cadets_e3 [theia_e3.dump]=theia_e3
  [optc_h501.dump]=optc_501 [optc_h201.dump]=optc_201 [optc_h051.dump]=optc_051)
for f in cadets_e3.dump theia_e3.dump optc_h501.dump optc_h201.dump optc_h051.dump; do
  su - pgsql -c "PATH=/root/miniconda3/envs/pids/bin:\$PATH \
    pg_restore -h 127.0.0.1 -p 5432 -U pgsql -d ${MAP[$f]} \
    --no-owner --no-privileges -j 4 /root/PIDSMaker/data/$f"
done
```

### 1.7 恢复 artifacts（**核心节省时间**点）

```bash
mkdir -p /root/PIDSMaker/artifacts
rsync -a /jizhicfs/PIDSMaker_root/artifacts/ /root/PIDSMaker/artifacts/
```

`artifacts/` 包含已训好的 word2vec、节点 embedding、gnn_models（`model_epoch_*`）等。**PIDSMaker 用 cfg 内容算 hash 索引产物**——只要 cfg 与之前完全一致（我们的补丁 + orthrus.yml num_workers=32 这些都已落到 code_files 里），就会**直接命中缓存**，跳过对应阶段。

### 1.8 续跑

```bash
cd /root/PIDSMaker
source /root/miniconda3/etc/profile.d/conda.sh && conda activate pids
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8

# 看上次跑到哪
bash check_progress.sh

# 启动后台任务（按需）
# - 如果 flash 没跑完所有 5 数据集训练 + 6 transfer：
nohup bash run_all_transfer_experiments.sh > artifacts/run_all.log 2>&1 &
# - 如果只想跑 orthrus（GPU1）：
CUDA_VISIBLE_DEVICES=1 nohup bash run_orthrus_only.sh > artifacts/run_orthrus.log 2>&1 &
# - 如果只想跑 velox（GPU0 与 flash 共用）：
CUDA_VISIBLE_DEVICES=0 nohup bash run_velox_only.sh > artifacts/run_velox.log 2>&1 &
```

> **续跑会自动跳过已完成的 task**（`run_all_transfer_experiments.sh` 当前实现里没显式 resume 检查，但每个 train task 内部 PIDSMaker 会自动从 cfg-hash artifacts 命中——所以 train 阶段秒过；transfer 阶段会重新跑 inference，不耗时）。

### 1.9 重启持久化同步（保持 sync 到 jizhicfs）

```bash
nohup bash /root/PIDSMaker/scripts_persist/sync_to_persist.sh \
    > /jizhicfs/PIDSMaker_root/logs/sync_daemon.log 2>&1 &
```

---

## 2. 关键路径与配置（速查表）

| 项目 | 路径/值 |
|---|---|
| Conda env | `/root/miniconda3/envs/pids` |
| PIDSMaker | `/root/PIDSMaker`（v2.1.0 + PUDA 补丁） |
| puda-transfer | `/root/puda-transfer`（公开仓库） |
| PostgreSQL data | `/home/pgsql/pgdata`（user=pgsql, trust 模式） |
| Postgres URL | `host=127.0.0.1 port=5432 user=pgsql password="" dbname=<see map>` |
| Dataset → DB 映射 | `cadets_e3 → cadets_e3`, `theia_e3 → theia_e3`, `optc_h501 → optc_501`, `optc_h201 → optc_201`, `optc_h051 → optc_051` |
| artifacts 根 | `/root/PIDSMaker/artifacts` |
| 持久化备份 | `/jizhicfs/PIDSMaker_root/artifacts` |
| 最终结果 CSV | `artifacts/transfer_results/all_results.csv` |

## 3. 常见问题排查

| 症状 | 原因 / 解决 |
|---|---|
| `pg_restore: error: unsupported version (1.16)` | conda PG 版本是 16，必须升 17：`conda install -c conda-forge postgresql=17` |
| `initdb: error: cannot be run as root` | 用 `pgsql` 普通用户跑（见 1.5） |
| `pyg_lib` / `torch_sparse` 加载失败 GLIBC_2.29 | 系统 glibc 太老，PyG 2.6.0 自动 fallback，可忽略 |
| `OpenBLAS warning: NUM_THREADS exceeded` | 设 `export OMP_NUM_THREADS=8` 等 |
| `transfer_inference.py: No source model for X` | source 数据集还没训练（gnn_models/ 不存在），先跑 Phase 1 |
| flash word2vec 巨慢 | 这是 flash detector 自己的 word2vec，已优化（PositionalEncoder 提循环外，feat_inference_flash.py） |
| orthrus word2vec 巨慢 | 必须保留 `orthrus.yml: word2vec.num_workers: 32`（我们改过的版本） |

## 4. PUDA 实验设计（提醒）

55 个 task = 5 detector × (5 train + 6 transfer)，但本次实际跑的是 3 个 detector（flash + orthrus + velox），共 33 task：

```
DETECTORS = flash, orthrus, velox      # magic & kairos 已被排除
DATASETS  = CADETS_E3, THEIA_E3, optc_h501, optc_h201, optc_h051

Transfer scenarios:
  S3: CADETS_E3 → THEIA_E3
  S4: THEIA_E3  → CADETS_E3
  S5: THEIA_E3  → optc_h501
  S6: optc_h501 → THEIA_E3
  S7: optc_h201 → CADETS_E3
  S8: CADETS_E3 → optc_h051
```

每个 transfer 写 1 行到 `all_results.csv`：`system,scenario,source,target,TP,TN,FP,FN,MCC,F1,ADP`。

## 5. 联系信息 / 当前状态时点

最后一次确认快照：见 `/jizhicfs/PIDSMaker_root/logs/sync.log` 末尾。
首次完整 rsync：2026-05-28 16:50（55 GB）。
持续同步：每 5 分钟。
Deadline watcher：触发于 2026-05-29 10:00（容器 11:00 到期前 1 小时）。
