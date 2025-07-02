# MvHo-IB: Multi-View Higher-Order Information Bottleneck for Brain Disorder Diagnosis

Code for "MvHo-IB: Multi-View Higher-Order Information Bottleneck for Brain Disorder Diagnosis", MICCAI-25.

## 🚀 Key Features

- **Multi-view Learning**: Simultaneously leverages pairwise connections and higher-order interactions in brain networks
- **Information Bottleneck Principle**: Improves model generalization through Rényi entropy regularization
- **Higher-order Interaction Modeling**: Quantifies synergistic/redundant relationships among triplet brain regions using O-information
- **End-to-end Training**: Unified framework for simultaneous optimization of feature extraction and fusion from two views
- **Flexible Configuration**: Supports multiple datasets and classification modes

## 📋 Method Overview

![MvHo-IB Framework Overview](images/ovreview.png)

The MvHo-IB framework integrates two complementary views of brain connectivity data through a unified information bottleneck approach, enabling robust brain disorder diagnosis through multi-view learning.

## 📊 Supported Datasets

- **UCLA**
- **ADNI**
- **EOEC**

## 🛠️ Requirements

- Python 3.8+
- PyTorch 1.10+
- PyTorch Geometric
- NumPy, scikit-learn, PyYAML, tqdm

## 📦 Installation

```bash
git clone https://github.com/zky04/MvHo-IB.git
cd MvHo-IB
pip install -r requirements.txt
pip install torch-geometric
```

## 🎯 Quick Start

### 1. Prepare Data
Place your dataset files in the `data/` directory:
- `x1_[dataset].pt`: First view data (e.g., functional connectivity)
- `x2_o_[dataset].pt`: Second view data (e.g., higher-order interactions)

Example structure:
```
data/
├── x1_ucla.pt
├── x2_o_ucla.pt
├── x1_adni.pt
└── x2_o_adni.pt
```

### 2. Configure Experiment
Edit `config.yaml` to set dataset and hyperparameters:

### 3. Run Training
```bash
python main.py
```

## 📁 Project Structure

```
MvHo-IB/
├── main.py              # Main entry point
├── config.yaml          # Configuration file
├── requirements.txt     # Dependencies
├── src/                 # Source code
│   ├── models/          # Neural network models
│   │   ├── gin_model.py        # GIN network
│   │   ├── brain3dcnn.py       # Brain3DCNN
│   │   └── fusion_model.py     # Feature fusion
│   ├── data/            # Data processing
│   │   ├── dataset.py          # Dataset classes
│   │   └── data_loader.py      # Data loading
│   ├── utils/           # Utility functions
│   │   ├── info_bottleneck.py  # Information bottleneck
│   │   ├── evaluator.py        # Evaluation metrics
│   │   └── config_utils.py     # Configuration processing
│   └── trainer/         # Trainer
│       └── trainer.py          # Training logic
└── experiments/         # Experimental results
```
