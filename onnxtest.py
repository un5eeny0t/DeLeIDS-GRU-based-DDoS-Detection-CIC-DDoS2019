import onnxruntime as ort
import numpy as np
import yaml     
from preprocess import preprocess_onnx, get_numeric_columns
import pandas as pd
import time


# --- Load Configuration from YAML ---
with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

# Get paths from config
csv_file = config.get('csv_file', 'data/CICDdos2019/csv/01-12/TFTP.csv')
onnx_model_path = config.get('onnx_model_path', 'gru_tftp_model.onnx')
batch_size = config.get('batch_size', 64)

print(f"Loading test data from: {csv_file}")

# Load test data from CSV (using a sample for testing)
# In production, you'd want a separate test CSV file
print("Loading and preprocessing test data...")
test_df = pd.read_csv(csv_file, nrows=10000, low_memory=False)  # Load sample for testing
test_loader = preprocess_onnx(test_df, batch_size=batch_size, scaler_save_path='scaler.pkl')

print(f"Test data loaded: {len(test_df)} samples")

# Load your ONNX model
print(f"Loading ONNX model from: {onnx_model_path}")
onnx_session = ort.InferenceSession(onnx_model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

# Print which provider is actually being used
print(f"Using provider: {onnx_session.get_providers()}")

# Get model input and output names
input_name = onnx_session.get_inputs()[0].name
output_name = onnx_session.get_outputs()[0].name

print(f"Input name: {input_name}, Output name: {output_name}")

correct = 0
total = 0

# Warm-up (Optional)
print("Warming up model...")
dummy_batch = next(iter(test_loader))[0].numpy().astype(np.float32)
# GRU expects (batch_size, sequence_length, input_size)
dummy_batch = dummy_batch.reshape(dummy_batch.shape[0], 1, dummy_batch.shape[1])
onnx_session.run([output_name], {input_name: dummy_batch})

print("Starting timed inference...")
start_time = time.perf_counter()

for batch_data, labels in test_loader:
    # Reshape for GRU: (batch_size, sequence_length=1, input_size)
    input_data = batch_data.numpy().astype(np.float32)
    input_data = input_data.reshape(input_data.shape[0], 1, input_data.shape[1])

    # Run inference
    result = onnx_session.run([output_name], {input_name: input_data})[0]
    
    # Binary classification: threshold at 0.5
    predicted = (result > 0.5).astype(np.int32).flatten()
    labels_np = labels.numpy().astype(np.int32)
    
    total += labels.size(0)
    correct += (predicted == labels_np).sum().item()

end_time = time.perf_counter()
total_time = end_time - start_time

print(f"\n--- Performance Report ---")
print(f"Total time for {len(test_loader)} batches: {total_time:.4f} seconds")
print(f"Average time per batch: {(total_time / len(test_loader)) * 1000:.2f} ms")
print(f"Throughput: {total / total_time:.2f} samples/sec")
print(f"ONNX Model Accuracy: {100 * correct / total:.2f}%")
