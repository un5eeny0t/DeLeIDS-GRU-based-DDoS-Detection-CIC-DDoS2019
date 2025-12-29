# DeLeIDS – GRU-based DDoS Detection (CIC-DDoS2019)

DeLeIDS is a deep learning–based Intrusion Detection System (IDS) that uses a **GRU (Gated Recurrent Unit)** model to detect **TFTP-based DDoS attacks** from the **CIC-DDoS2019 dataset**.  
The project is designed to run on **low-RAM systems** using **chunked CSV processing** and supports **experiment tracking with Weights & Biases (W&B)** and **deployment-ready ONNX export**.

---

## 🚀 Features

- **GRU-based binary classification** (BENIGN vs TFTP DDoS)
- **Chunk-based CSV processing** (handles 9GB+ files efficiently)
- **Class-weighted loss function** to handle severe class imbalance (0.21% BENIGN vs 99.79% ATTACK)
- **Comprehensive evaluation metrics** (Accuracy, Precision, Recall, F1-Score, Confusion Matrix)
- **CPU-friendly** (no GPU required)
- **W&B integration** for training and evaluation visualization
- **ONNX export** for cross-platform deployment
- **Automatic preprocessing** with feature scaling and cleaning

---

## ⚠️ Important: Class Imbalance

The CIC-DDoS2019 TFTP dataset has **severe class imbalance**:
- **BENIGN**: ~0.21% of samples
- **ATTACK**: ~99.79% of samples

This means models can achieve >99% accuracy by simply predicting the majority class. We address this with:
- **Class-weighted Binary Cross-Entropy loss**
- **Comprehensive metrics** (F1-Score, Recall, Precision) instead of relying solely on accuracy
- **Automatic class distribution reporting** per chunk

See [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md) for detailed interpretation guidelines.

---

## 🧠 Model Architecture

- **Input**: `(batch_size, sequence_length=1, num_features)`
- **GRU layers**: 2 layers with 64 hidden units
- **Fully connected layer**: Binary classification output
- **Loss**: Weighted Binary Cross-Entropy (automatically balanced per chunk)
- **Output**: Probability of attack (sigmoid activation)
- **Optimizer**: Adam (learning rate: 0.001)

---

## 📂 Dataset

- **Dataset**: [CIC-DDoS2019](https://www.unb.ca/cic/datasets/ddos-2019.html)
- **Attack Type**: TFTP-based DDoS
- **Format**: CSV (flow-based features)
- **Scale**: Millions of records
- **Location**: `data/CICDdos2019/csv/01-12/TFTP.csv`


## 📁 Project Structure

```
DeLeIDS/
├── data/
│   └── CICDdos2019/
│       └── csv/
│           ├── 01-12/
│           │   └── TFTP.csv                # Main training dataset
│           └── 03-11/                     # Additional data
│
├── model.py                                # GRUNet and IDSModel definitions
├── train_gru_cicddos2019.py                # Main training script
├── preprocess.py                           # Data preprocessing and data loaders
├── convert.py                              # Model to ONNX conversion
├── onnxtest.py                             # ONNX model testing
├── config.yml                              # Configuration file
├── headerfind.py                           # Utility to inspect CSV headers
│
├── REPORT.md                               # Detailed experiment report
├── EVALUATION_GUIDE.md                     # Guide for interpreting results
├── README.md                               # This file
├── pyproject.toml                          # Project dependencies (uv)
│
├── wandb/                                  # Auto-created W&B logs
└── gru_tftp_model.pth                     # Saved model weights (after training)
```

---

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd DeLeIDS
   ```

2. **Install dependencies** (using uv):
   ```bash
   uv sync
   ```

   Or using pip:
   ```bash
   pip install torch pandas numpy scikit-learn wandb onnxruntime pyyaml
   ```

3. **Download CIC-DDoS2019 dataset**:
   - Download from [CIC-DDoS2019 website](https://www.unb.ca/cic/datasets/ddos-2019.html)
   - Extract `TFTP.csv` to `data/CICDdos2019/csv/01-12/TFTP.csv`

4. **Login to Weights & Biases** (optional):
   ```bash
   wandb login
   ```

---

## 🚀 Usage

### Training

Train the GRU model on the TFTP dataset:

```bash
# Using uv
uv run train_gru_cicddos2019.py

# Or directly with Python
python train_gru_cicddos2019.py
```

The script will:
- Process data in chunks of 500,000 samples
- Apply class-weighted loss automatically
- Report comprehensive metrics per chunk
- Save model to `gru_tftp_model.pth`
- Log metrics to W&B (if configured)

### Model Conversion to ONNX

Convert the trained model to ONNX format:

```bash
uv run convert.py
```

This will:
- Load the trained model
- Export to `gru_tftp_model.onnx`
- Support dynamic batch sizes

### Testing ONNX Model

Test the ONNX model performance:

```bash
uv run onnxtest.py
```

---

## 📊 Evaluation Metrics

The training script reports:

- **Accuracy**: Overall classification correctness
- **Precision**: When model predicts a class, how often is it correct?
- **Recall**: What percentage of each class is correctly identified?
- **F1-Score**: Harmonic mean of precision and recall (best for imbalanced data)
- **Confusion Matrix**: Detailed breakdown (TN, FP, FN, TP)

**Important**: For imbalanced data, focus on **F1-Score** and **Recall** rather than Accuracy alone.

See [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md) for detailed interpretation.

---

## 📝 Configuration

Edit `config.yml` to customize:

- Model architecture (hidden_size, num_layers)
- Training parameters (batch_size, learning_rate, epochs)
- Data paths
- Chunk size

---

## 📚 Documentation

- **[REPORT.md](REPORT.md)**: Detailed experiment report with findings
- **[EVALUATION_GUIDE.md](EVALUATION_GUIDE.md)**: Guide for interpreting results on imbalanced data

---

## 🔍 Key Findings

1. **Class Imbalance**: The dataset has severe imbalance (0.21% BENIGN vs 99.79% ATTACK)
2. **Weighted Loss**: Class-weighted loss significantly improves minority class detection
3. **Metrics Matter**: Accuracy alone is misleading; F1-Score and Recall are more reliable
4. **Chunk-Based Training**: Efficiently handles large datasets on commodity hardware

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---



## 🙏 Acknowledgments

- [CIC-DDoS2019 Dataset](https://www.unb.ca/cic/datasets/ddos-2019.html) by Canadian Institute for Cybersecurity
- PyTorch team
- Weights & Biases for experiment tracking

