import torch.nn as nn
import torch

class IDSModel(nn.Module):
    """Feedforward neural network for multi-class classification (CICDDoS2019)"""
    def __init__(self, input_features, num_classes):
        super(IDSModel, self).__init__()
        self.fc1 = nn.Linear(input_features, 256)
        self.batch_norm1 = nn.BatchNorm1d(256)
        self.activation1 = nn.GELU()
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, 128)
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.activation2 = nn.GELU()
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(128, num_classes)
    

    def forward(self, x):
        x = self.fc1(x)
        x = self.batch_norm1(x)  
        x = self.activation1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.batch_norm2(x)  
        x = self.activation2(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        return x


class GRUNet(nn.Module):
    """GRU-based binary classifier for CIC-DDoS2019"""
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(GRUNet, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]  # last time step
        out = torch.sigmoid(self.fc(out))
        return out