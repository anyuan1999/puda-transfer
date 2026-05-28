# PIDSMaker Cross-Domain Transfer (PUDA Experiments)

基于 PIDSMaker v2.1.0 的跨域迁移实验框架。

## 快速开始 (H20 双卡机器)

```bash
# 1. Clone 原始 PIDSMaker
git clone https://github.com/ubc-provenance/PIDSMaker.git
cd PIDSMaker
git checkout tags/2.1.0 -b v2.1.0

# 2. 应用我们的迁移补丁
cp -r ../puda-transfer/* .

# 3. 配置环境
cp .env.local .env
# 编辑 .env: INPUT_DIR=./data, ARTIFACTS_DIR=./artifacts

# 4. 启动容器 (有 GPU 不需要 --cpu)
docker compose -p postgres -f compose-postgres.yml up -d --build
docker compose -f compose-pidsmaker.yml up -d --build

# 5. 下载数据集
docker exec pidsmaker-pids bash -c "source activate pids && cd /home/pids && \
  ./download_datasets.sh cadets_e3,theia_e3,optc_h051,optc_h201,optc_h501 YOUR_TOKEN"

# 6. 加载数据库
docker exec postgres bash -c "./scripts/load_dumps.sh"

# 7. 运行全部迁移实验
docker exec -u pids pidsmaker-pids bash -c "source activate pids && cd /home/pids && \
  bash run_all_transfer_experiments.sh"
```

## 文件说明

| 文件 | 作用 |
|------|------|
| `transfer_inference.py` | 核心迁移推理脚本 |
| `run_all_transfer_experiments.sh` | 批量运行所有检测器×场景 |
| `patches/training_loop.patch` | 启用模型保存 |
| `patches/data_utils.patch` | 修复 save_model TGN 兼容性 |
| `config/magic_transfer.yml` | MAGIC 大数据集配置 |

## 迁移实验流程

```
Phase 1: python pidsmaker/main.py <detector> <source_dataset> [--cpu]
Phase 2: python transfer_inference.py <detector> <source> <target> [--cpu]
```

## 注意事项 (H20 + CUDA)

RTX H20 使用 Hopper 架构，需要:
- PyTorch 2.x + CUDA 12.1+
- 修改 Dockerfile 中的 PyTorch 安装行 (见 patches/Dockerfile.h20.patch)
