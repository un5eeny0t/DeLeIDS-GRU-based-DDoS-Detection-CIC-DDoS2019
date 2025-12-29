import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset


def map_labels_to_binary(df, label_column='Label'):
    """
    Map labels to binary classification: 0 for BENIGN, 1 for all other labels.
    
    Args:
        df: DataFrame with a label column
        label_column: Name of the label column (default: 'Label')
    
    Returns:
        Series with binary labels (0 for BENIGN, 1 for others)
    """
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in DataFrame")
    
    y = df[label_column].copy()
    # BENIGN -> 0, everything else -> 1
    y = (y != 'BENIGN').astype(np.float32)
    
    return y


def preprocess_chunk(df, scaler, numeric_cols, label_column='Label'):
    """
    Preprocess a chunk of data: clean, scale features, and map labels.
    
    Args:
        df: DataFrame chunk to preprocess
        scaler: StandardScaler instance (will be partial_fit)
        numeric_cols: List of numeric column names
        label_column: Name of the label column (default: 'Label')
    
    Returns:
        X_scaled: Scaled feature array
        y: Binary label Series
    """
    df.columns = df.columns.str.strip()  # Strip spaces
    
    if label_column not in df.columns:
        print(f"[-] Label column '{label_column}' not found, skipping chunk")
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
    
    # Map labels to binary
    y = map_labels_to_binary(df, label_column)
    
    return X_scaled, y


def create_data_loaders(csv_file, chunk_size=500000, test_size=0.2, batch_size=64, 
                        label_column='Label', random_state=42, stratify=True):
    """
    Create PyTorch DataLoaders from CSV file using chunk-based processing.
    
    Args:
        csv_file: Path to CSV file
        chunk_size: Number of rows to process per chunk
        test_size: Proportion of data for validation (default: 0.2)
        batch_size: Batch size for DataLoaders (default: 64)
        label_column: Name of the label column (default: 'Label')
        random_state: Random seed for train/test split
        stratify: Whether to use stratified split (default: True)
    
    Yields:
        train_loader: PyTorch DataLoader for training
        val_loader: PyTorch DataLoader for validation
        chunk_idx: Current chunk index
    """
    # Read CSV in chunks
    csv_iter = pd.read_csv(csv_file, chunksize=chunk_size, low_memory=False)
    
    # Determine numeric columns from first chunk
    first_chunk = next(csv_iter)
    first_chunk.columns = first_chunk.columns.str.strip()
    numeric_cols = first_chunk.select_dtypes(include=np.number).columns.tolist()
    
    # Reset iterator
    csv_iter = pd.read_csv(csv_file, chunksize=chunk_size, low_memory=False)
    
    scaler = StandardScaler()
    
    for chunk_idx, df in enumerate(csv_iter, start=1):
        print(f"[+] Processing chunk {chunk_idx}")
        
        X, y = preprocess_chunk(df, scaler, numeric_cols, label_column)
        if X is None or y is None or len(y.unique()) < 2:
            print("[-] Skipping chunk (insufficient classes)")
            continue
        
        # Split train/validation
        try:
            if stratify:
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=test_size, stratify=y, random_state=random_state
                )
            else:
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=test_size, random_state=random_state
                )
        except ValueError:
            # Fallback: non-stratified split if class imbalance occurs
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
        
        # Create PyTorch datasets
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train.values, dtype=torch.float32)
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val.values, dtype=torch.float32)
        )
        
        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        yield train_loader, val_loader, chunk_idx


def get_numeric_columns(csv_file, nrows=1000):
    """
    Get list of numeric columns from CSV file.
    
    Args:
        csv_file: Path to CSV file
        nrows: Number of rows to read for column detection (default: 1000)
    
    Returns:
        List of numeric column names
    """
    df = pd.read_csv(csv_file, nrows=nrows, low_memory=False)
    df.columns = df.columns.str.strip()
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    return numeric_cols


def preprocess_onnx(data, batch_size=64, scaler_save_path='scaler.pkl', label_column='Label'):
    """
    Preprocess data for ONNX inference testing.
    
    Args:
        data: numpy array or DataFrame with features and labels
        batch_size: Batch size for DataLoader
        scaler_save_path: Path to saved scaler (if None, will create new one)
        label_column: Name of label column if data is DataFrame
    
    Returns:
        DataLoader with preprocessed data
    """
    import pickle
    from sklearn.preprocessing import StandardScaler
    
    # Handle DataFrame input
    if isinstance(data, pd.DataFrame):
        df = data.copy()
        df.columns = df.columns.str.strip()
        
        if label_column not in df.columns:
            raise ValueError(f"Label column '{label_column}' not found")
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if label_column in numeric_cols:
            numeric_cols.remove(label_column)
        
        X = df[numeric_cols].copy()
        y = map_labels_to_binary(df, label_column)
        
        # Preprocess features
        X = X.apply(pd.to_numeric, errors='coerce')
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(0, inplace=True)
        X = X.clip(lower=-1e9, upper=1e9)
        
        # Load or create scaler
        try:
            with open(scaler_save_path, 'rb') as f:
                scaler = pickle.load(f)
        except FileNotFoundError:
            scaler = StandardScaler()
            scaler.fit(X)
            with open(scaler_save_path, 'wb') as f:
                pickle.dump(scaler, f)
        
        X_scaled = scaler.transform(X)
        
    else:
        # Handle numpy array input (assume last column is label)
        X = data[:, :-1]
        y = data[:, -1]
        
        # Load or create scaler
        try:
            with open(scaler_save_path, 'rb') as f:
                scaler = pickle.load(f)
        except FileNotFoundError:
            scaler = StandardScaler()
            scaler.fit(X)
            with open(scaler_save_path, 'wb') as f:
                pickle.dump(scaler, f)
        
        X_scaled = scaler.transform(X)
        y = pd.Series(y).astype(np.float32)
    
    # Create DataLoader
    dataset = TensorDataset(
        torch.tensor(X_scaled, dtype=torch.float32),
        torch.tensor(y.values if isinstance(y, pd.Series) else y, dtype=torch.float32)
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    return loader

