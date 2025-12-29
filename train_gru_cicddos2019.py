import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import wandb
from model import GRUNet

# -------------------------------
# GRU model is now imported from model.py
# -------------------------------
# Preprocess each chunk
# -------------------------------
def preprocess_chunk(df, scaler, numeric_cols):
    df.columns = df.columns.str.strip()  # Strip spaces

    if "Label" not in df.columns:
        print("[-] Label column not found, skipping chunk")
        return None, None

    # Keep only numeric columns
    X = df.reindex(columns=numeric_cols, fill_value=0)

    # Force numeric, coerce errors to NaN
    X = X.apply(pd.to_numeric, errors='coerce')

    # Clean invalid values
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True)
    X = X.clip(lower=-1e9, upper=1e9)

    # Fit/transform scaler
    scaler.partial_fit(X)
    X_scaled = scaler.transform(X)

    # Labels: 0 for BENIGN, 1 for all other (non-benign) labels
    y = df["Label"].copy()
    y = (y != 'BENIGN').astype(np.float32)  # BENIGN -> 0, everything else -> 1

    return X_scaled, y

# -------------------------------
# Training loop per chunk
# -------------------------------
def train_chunk(model, criterion, optimizer, X_train, y_train, X_val, y_val, device, epochs=1, batch_size=64):
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(y_train.values, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                torch.tensor(y_val.values, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            xb = xb.unsqueeze(1)  # add sequence dimension
            optimizer.zero_grad()
            preds = model(xb).squeeze()
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        correct = total = 0
        val_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                xb = xb.unsqueeze(1)
                preds = model(xb).squeeze()
                loss = criterion(preds, yb)
                val_loss += loss.item()
                predicted = (preds > 0.5).float()
                total += yb.size(0)
                correct += (predicted == yb).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(yb.cpu().numpy())
        
        # Calculate comprehensive metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        val_acc = correct / total
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        cm = confusion_matrix(all_labels, all_preds)
        
    return loss.item(), val_loss / len(val_loader), val_acc, precision, recall, f1, cm

# -------------------------------
# Main function
# -------------------------------
def main():
    wandb.init(project="CIC-DDoS2019-GRU", name="gru-tftp-chunked")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Using device: {device}")

    csv_file = "data/CICDdos2019/csv/01-12/TFTP.csv"
    chunk_size = 500000

    # Read CSV in chunks
    csv_iter = pd.read_csv(csv_file, chunksize=chunk_size, low_memory=False)
    
    # Determine numeric columns from first chunk
    first_chunk = next(csv_iter)
    first_chunk.columns = first_chunk.columns.str.strip()
    numeric_cols = first_chunk.select_dtypes(include=np.number).columns.tolist()
    csv_iter = pd.read_csv(csv_file, chunksize=chunk_size, low_memory=False)  # reset iterator

    scaler = StandardScaler()
    input_size = len(numeric_cols)

    model = GRUNet(input_size).to(device)
    
    # Use weighted loss to handle class imbalance
    # Will be updated per chunk based on actual class distribution
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs_per_chunk = 1

    for chunk_idx, df in enumerate(csv_iter, start=1):
        print(f"[+] Processing chunk {chunk_idx}")

        X, y = preprocess_chunk(df, scaler, numeric_cols)
        if X is None or y is None or len(y.unique()) < 2:
            print("[-] Skipping chunk (insufficient classes)")
            continue

        # Report class distribution
        benign_count = (y == 0).sum()
        attack_count = (y == 1).sum()
        total_samples = len(y)
        benign_pct = (benign_count / total_samples) * 100
        attack_pct = (attack_count / total_samples) * 100
        print(f"    Class distribution: BENIGN={benign_count} ({benign_pct:.2f}%), "
              f"ATTACK={attack_count} ({attack_pct:.2f}%)")
        
        # Warn about severe imbalance
        if benign_pct < 1 or attack_pct < 1:
            print(f"    ⚠️  WARNING: Severe class imbalance detected!")
        elif abs(benign_pct - attack_pct) > 80:
            print(f"    ⚠️  WARNING: High class imbalance - accuracy may be misleading!")
        
        # Calculate class weights for balanced loss
        if benign_count > 0 and attack_count > 0:
            total = benign_count + attack_count
            weight_benign = total / (2 * benign_count)
            weight_attack = total / (2 * attack_count)
            # Create weighted BCELoss function
            base_criterion = nn.BCELoss(reduction='none')
            # Convert to tensors for use in loss function
            w_benign_tensor = torch.tensor(weight_benign, dtype=torch.float32).to(device)
            w_attack_tensor = torch.tensor(weight_attack, dtype=torch.float32).to(device)
            
            def weighted_criterion(pred, target):
                loss = base_criterion(pred, target)
                # Apply weights: weight_benign for class 0, weight_attack for class 1
                weights = torch.where(target == 0, w_benign_tensor, w_attack_tensor)
                return (loss * weights).mean()
            
            criterion = weighted_criterion
            print(f"    Using class weights: BENIGN={weight_benign:.3f}, ATTACK={weight_attack:.3f}")
        else:
            criterion = nn.BCELoss()

        # Split train/validation
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42
            )
        except ValueError:
            # fallback: non-stratified split if class imbalance occurs
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

        loss, val_loss, val_acc, precision, recall, f1, cm = train_chunk(
            model, criterion, optimizer, X_train, y_train, X_val, y_val, device,
            epochs=epochs_per_chunk)
        
        print(f"    Train loss: {loss:.4f}")
        print(f"    Val loss:   {val_loss:.4f}")
        print(f"    Val acc:    {val_acc:.4f}")
        print(f"    Precision:  {precision:.4f}")
        print(f"    Recall:     {recall:.4f}")
        print(f"    F1-score:   {f1:.4f}")
        print(f"    Confusion Matrix:")
        print(f"      [[TN={cm[0,0]:6d}, FP={cm[0,1]:6d}]")
        print(f"       [FN={cm[1,0]:6d}, TP={cm[1,1]:6d}]]")

        # Log metrics
        wandb.log({
            "chunk": chunk_idx,
            "train_loss": loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_precision": precision,
            "val_recall": recall,
            "val_f1": f1,
            "benign_pct": benign_pct,
            "attack_pct": attack_pct
        })

    # Save final model
    torch.save(model.state_dict(), "gru_tftp_model.pth")
    print("[+] Training complete. Model saved.")

if __name__ == "__main__":
    main()
