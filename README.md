# AMPLoc

AMPLoc 是一个用于 lncRNA 亚细胞定位预测的多模态融合系统。当前仓库的主链路以 `main_run.py` 为入口，采用“监督预热 + PPO 微调”的两阶段训练流程；模型会根据样本状态选择可用通道，再送入 policy-guided bottleneck fusion 完成分类。

## 发布范围

这个仓库面向 GitHub 发布时，建议只保留下面这些内容：

- 主训练代码：`main_run.py`、`main_train.py`、`src/`
- 数据集 1 的核心数据：`data/rna_data.csv`、`data/structures.csv`、`data/rigorous_splits/dataset1_*.csv`、`data/rpi_data/`
- 主配置文件：`configs/main_config.yaml`
- 环境文件：`environment.yml`、`requirements.txt`
- 核心模型权重：`pretrained/channel_agent.pth`

建议不要直接提交到主仓库的内容：

- 大体积派生缓存：`data/cgr_features/`、`data/processed/`、`data/processed_structures*/`、`data/raw/`
- 大型预训练底座：`pretrained/DNABERT-2-117M/`、`pretrained/RNA-FM/`
- 训练输出与临时实验产物：`outputs/`、`temp_test_files/`
- dataset2 及其它一次性实验数据

如果某些派生数据对复现是必要的，但体积太大，建议放到 GitHub Release、对象存储或单独的数据包中；README 里只保留下载方式和重建顺序。

## 目录结构

```text
AMPLoc/
├── main_run.py
├── main_train.py
├── configs/
│   └── main_config.yaml
├── data/
│   ├── rna_data.csv
│   ├── structures.csv
│   ├── rigorous_splits/
│   └── rpi_data/
├── pretrained/
│   └── channel_agent.pth
├── scripts/
│   ├── data_prep/
│   │   ├── generate_structure_csv_dataset1.py
│   │   ├── generate_structure_csv_dataset2.py
│   │   ├── build_rigorous_splits.py
│   │   ├── inspect_train_val_split.py
│   │   └── split_validation_dataset.py
│   └── analysis/
│       └── amploc/
├── src/
└── environment.yml
```

## 核心脚本

- `scripts/data_prep/generate_structure_csv_dataset1.py`：从 LinearFold 结构文件生成 dataset1 的 `structures.csv`。
- `scripts/data_prep/generate_structure_csv_dataset2.py`：dataset2 的结构 CSV 生成脚本，保留给历史实验或本地复现实验使用。
- `scripts/data_prep/build_rigorous_splits.py`：生成严格划分文件，包含 held-out test 和 development split。
- `scripts/data_prep/inspect_train_val_split.py`：检查训练/验证切分比例和标签分布。
- `scripts/data_prep/split_validation_dataset.py`：把验证集进一步拆成训练/测试子集。
- `scripts/analysis/amploc/`：训练历史、消融、热图和 RBP 重要性分析脚本。

## 环境安装

推荐使用 Conda：

```bash
conda env create -f environment.yml
conda activate amploc
```

如果你更习惯 pip，也可以先创建虚拟环境，再安装：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如果你的机器使用的是特定 CUDA 版本，PyTorch 和 PyG 相关轮子可能需要按本地环境额外调整；`environment.yml` 提供的是发布版的统一入口，方便多数用户快速复现。

## 数据准备

默认主实验以 dataset1 为准。运行前请确认下面文件存在：

- `data/rna_data.csv`
- `data/structures.csv`
- `data/rigorous_splits/dataset1_development.csv`
- `data/rigorous_splits/dataset1_held_out_test.csv`
- `data/rpi_data/rpi_scores.csv`
- `data/rpi_data/rpi_scores_GGCN.csv`
- `data/rpi_data/rpi_scores_LPI.csv`
- `data/rpi_data/rpi_scores_SVM.csv`
- `data/rpi_data/rbp_locations.csv`
- `pretrained/channel_agent.pth`

如果你只有原始 RNA 数据而没有 `structures.csv`，可以先用结构生成脚本重建；如果 `data/processed_structures_linearfold/` 这类结构缓存太大，不建议直接塞进 GitHub，推荐保留最终 `structures.csv` 作为发布件即可。

## 运行主训练

```bash
python main_run.py --config configs/main_config.yaml
```

如果需要指定输出目录：

```bash
python main_run.py --config configs/main_config.yaml --output_dir outputs/my_run
```

如果要固定 GPU：

```bash
CUDA_VISIBLE_DEVICES=0 python main_run.py --config configs/main_config.yaml
```

## 训练流程

1. `main_run.py` 读取配置并构建数据集、Tokenizer 和 DataLoader。
2. `main_train.py` 初始化 `DynamicFusionModel`、`PPOAgent` 和 `AgentInferenceManager`。
3. 第 1 阶段进行监督预热，先稳定融合主干。
4. 第 2 阶段进行 PPO 微调，让智能体学习更优的通道组合。
5. 验证阶段会记录 `Ave-F1`、`MaAUC`、`MiP`、`MiR`、`MiF`，并在达到目标区间时早停。

## 配置重点

主配置集中在 `configs/main_config.yaml`：

- `training.supervised_epochs` 和 `training.agent_epochs`：两阶段训练轮数。
- `training.target_min` / `training.target_max`：目标区间早停。
- `mbt_fuser.fusion_dim`、`num_layers`、`num_heads`、`num_fusion_tokens`：融合器参数。
- `ilocbert.seq_max_len`：DNABERT-2 输入长度，默认 512，用于控制显存开销。
- `channel_agent.enabled`：是否启用 PPO 智能体。
- `meta_architect.active_fusion_channels`：实际参与融合的通道列表，结构视图和 RPI source 会在运行时展开。

## 输出说明

每次运行一般会在 `outputs/<timestamp>/` 下生成：

- `config_run.yaml`
- `run.log`
- `best_model_supervised.pth`
- `best_model_rl.pth`

如果你打算公开仓库，建议把完整训练结果、缓存特征和临时分析物料都放在仓库外部，只保留最小可复现文件。

## 复现建议

如果你希望别人能尽量少配置就复现：

1. 只保留 dataset1 的核心 CSV、配置和必要权重。
2. 把大型预训练底座和派生缓存改成外部下载链接。
3. 在 README 中保留生成脚本、输入目录结构和重建顺序。
4. 为 dataset2 和临时实验单独开分支或单独数据包，不放进主发布仓库。

## 文档

- `docs/Project_Logic_Flow.md`
- `docs/Channel_Agent_Guide.md`
- `docs/RPI_Integration_Guide.md`

这些文档更适合用来理解当前主链路的设计和通道接法。