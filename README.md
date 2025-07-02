# MvHo-IB: Multi-View Higher-Order Information Bottleneck for Brain Disorder Diagnosis

Code for "MvHo-IB: Multi-View Higher-Order Information Bottleneck for Brain Disorder Diagnosis", MICCAI-25.

## 🚀 Key Features

- **Multi-view Learning**: Simultaneously leverages pairwise connections and higher-order interactions in brain networks
- **Information Bottleneck Principle**: Improves model generalization through Rényi entropy regularization
- **Higher-order Interaction Modeling**: Quantifies synergistic/redundant relationships among triplet brain regions using O-information
- **End-to-end Training**: Unified framework for simultaneous optimization of feature extraction and fusion from two views
- **Flexible Configuration**: Supports multiple datasets and classification modes

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
