# MvHo-IB: Multi-View Higher-Order Information Bottleneck for Brain Disorder Diagnosis

一个基于多视图学习和高阶信息交互的脑疾病诊断深度学习框架。

## 🚀 主要特性

- **多视图学习**: 同时利用脑网络的成对连接和高阶交互信息
- **信息瓶颈原理**: 通过Rényi熵正则化提高模型泛化能力  
- **高阶交互建模**: 基于O-information量化三元组脑区间的协同/冗余关系
- **端到端训练**: 统一的框架同时优化两个视图的特征提取和融合
- **灵活配置**: 支持多种数据集和分类模式

## 📊 支持的数据集

- **UCLA**: 105个脑区，精神分裂症诊断（正常 vs 精神分裂症）
- **ADNI**: 116个脑区，阿尔茨海默病诊断，支持多种分类模式：
  - 三分类：AD vs MCI vs NC
  - 二分类：AD vs NC / AD vs MCI / MCI vs NC
- **EOEC**: 116个脑区，静息态脑网络研究（眼开 vs 眼闭）

## 🛠️ 环境要求

- Python 3.8+
- PyTorch 1.10+
- PyTorch Geometric
- NumPy, scikit-learn, PyYAML, tqdm

## 📦 安装

```bash
git clone https://github.com/zky04/MvHo-IB.git
cd MvHo-IB
pip install -r requirements.txt
pip install torch-geometric
```

## 🎯 快速开始

### 1. 准备数据

### 2. 配置实验
编辑 `config.yaml` 设置数据集和超参数：

### 3. 运行训练
```bash
python main.py
```

## 📁 项目结构

```
MvHo-IB/
├── main.py              # 主程序入口
├── config.yaml          # 配置文件
├── requirements.txt     # 依赖包
├── src/                 # 源代码
│   ├── models/          # 神经网络模型
│   │   ├── gin_model.py        # GIN网络
│   │   ├── brain3dcnn.py       # Brain3DCNN
│   │   └── fusion_model.py     # 特征融合
│   ├── data/            # 数据处理
│   │   ├── dataset.py          # 数据集类
│   │   └── data_loader.py      # 数据加载
│   ├── utils/           # 工具函数
│   │   ├── info_bottleneck.py  # 信息瓶颈
│   │   ├── evaluator.py        # 评估指标
│   │   └── config_utils.py     # 配置处理
│   └── trainer/         # 训练器
│       └── trainer.py          # 训练逻辑
└── experiments/         # 实验结果
```
