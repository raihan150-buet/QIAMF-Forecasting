import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

class LoadForecastingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def prepare_data(csv_path, seq_len, pred_len, train_ratio=0.8, batch_size=32):
    # Load
    df = pd.read_excel(csv_path,sheet_name="Sheet1")
    if 'datetime' in df.columns:
        df = df.sort_values('datetime').drop(columns=['datetime'])
    
    data = df.values
    
    # Scale
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Create Sequences
    X, y = [], []
    # Last column is the target Column
    target_idx = -1 
    
    for i in range(len(data_scaled) - seq_len - pred_len + 1):
        X.append(data_scaled[i : i+seq_len])
        y.append(data_scaled[i+seq_len : i+seq_len+pred_len, target_idx])
        
    X = np.array(X)
    y = np.array(y)
    
    # Split
    train_size = int(len(X) * train_ratio)
    
    train_ds = LoadForecastingDataset(X[:train_size], y[:train_size])
    val_ds = LoadForecastingDataset(X[train_size:], y[train_size:])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, scaler, X.shape[2]