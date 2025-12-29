import torch
from model import GRUNet
import numpy as np
import yaml
import pandas as pd
from preprocess import get_numeric_columns

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

# Get model path from config or use default
model_path = config.get('model_path', 'gru_tftp_model.pth')
csv_file = config.get('csv_file', 'data/CICDdos2019/csv/01-12/TFTP.csv')

print(f"Loading model from: {model_path}")

# Get input size from CSV file
print("Determining input size from data...")
numeric_cols = get_numeric_columns(csv_file, nrows=1000)
input_size = len(numeric_cols)
print(f"Input size: {input_size}")

# Load model
hidden_size = config.get('hidden_size', 64)
num_layers = config.get('num_layers', 2)
model = GRUNet(input_size, hidden_size=hidden_size, num_layers=num_layers).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
print("Model loaded successfully")

print("\nExporting model to ONNX")
onnx_path = config.get('onnx_model_path', model_path.replace(".pth", ".onnx"))

# GRU expects input shape: (batch_size, sequence_length, input_size)
# For inference, we use sequence_length=1
dummy_input = torch.randn(size=(1, 1, input_size), device=device, dtype=torch.float32)

model.eval()
try:
    torch.onnx.export(
        model.to(device),
        dummy_input, onnx_path, verbose=False,
        input_names=['input'], output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'}, 
            'output': {0: 'batch_size'}
        },
        opset_version=11
    )
    print(f"Model exported to ONNX at: {onnx_path}")
except Exception as e:
    print(f"Error exporting model to ONNX: {e}")
    import traceback
    traceback.print_exc()