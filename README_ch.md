# MvHo-IB: Multi-View Higher-Order Information Bottleneck for Brain Disorder Diagnosis

<div align="center">

[English](README.md) | [中文](README_ch.md)

</div>

"MvHo-IB: Multi-View Higher-Order Information Bottleneck for Brain Disorder Diagnosis(https://arxiv.org/pdf/2507.02847)", MICCAI-25 的代码实现。

## 📋 方法概述

![MvHo-IB Framework Overview](images/ovreview.png)

我们的MvHo-IB框架通过多视图学习架构处理fMRI时间序列数据。该流水线估计功能连接模式并将其输入双处理路径。框架通过优化信息瓶颈目标学习联合表示 $Z = f_{\theta}(Z_1, Z_2)$：最大化 $I(Y; Z)$ 同时最小化 $I(X_1; Z_1) + I(X_2; Z_2)$。视图1通过互信息矩阵处理成对交互，视图2使用 $\mathcal{O}$-information 3D张量捕获高阶三元交互。

## 📊 支持的数据集

- **UCLA**
- **ADNI**
- **EOEC**

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
将数据集文件放在 `data/` 目录中：
- `x1_[dataset].pt`: 第一视图数据（如功能连接）
- `x2_o_[dataset].pt`: 第二视图数据（如高阶交互）

目录结构示例：
```
data/
├── x1_ucla.pt
├── x2_o_ucla.pt
├── x1_adni.pt
└── x2_o_adni.pt
```

### 2. 配置实验
编辑 `config.yaml` 设置数据集和超参数：

### 3. 运行训练
```bash
python main.py
```
### Brain3DCNN架构

![Brain3DCNN Architecture](images/3DBrainCNN.png)

Brain3DCNN架构是一种专门设计，利用结构脑网络的拓扑局部性来增强 $\mathcal{O}$-information 表示学习。这种分层设计捕获多尺度脑连接模式以实现准确的疾病诊断。

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
