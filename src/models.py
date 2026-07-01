import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import time

# ==========================================
# 1. PyTorch Multi-Layer Perceptron (MLP)
# ==========================================
class MLPClassifierNet(nn.Module):
    def __init__(self, input_dim):
        super(MLPClassifierNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.net(x)

def train_mlp(X_train, y_train, epochs=10, batch_size=128, lr=0.001, device='cpu'):
    input_dim = X_train.shape[1]
    model = MLPClassifierNet(input_dim).to(device)
    
    # Prepare data loaders
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    start_time = time.time()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch_x.size(0)
        
        epoch_loss /= len(X_train)
        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(f"  [MLP] Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")
            
    train_time = time.time() - start_time
    return model, train_time

def predict_mlp(model, X, device='cpu'):
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        outputs = model(X_tensor)
        preds = (outputs >= 0.5).int().cpu().numpy().flatten()
        probs = outputs.cpu().numpy().flatten()
    return preds, probs


# ==========================================
# 2. PyTorch Autoencoder (AE) for Anomaly Detection
# ==========================================
class AutoencoderNet(nn.Module):
    def __init__(self, input_dim):
        super(AutoencoderNet, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def train_autoencoder(X_train_normal, epochs=10, batch_size=128, lr=0.001, device='cpu'):
    input_dim = X_train_normal.shape[1]
    model = AutoencoderNet(input_dim).to(device)
    
    # Prepare data loaders
    X_tensor = torch.tensor(X_train_normal, dtype=torch.float32)
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    start_time = time.time()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_x, in loader:
            batch_x = batch_x.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_x) # Reconstruct inputs
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch_x.size(0)
            
        epoch_loss /= len(X_train_normal)
        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(f"  [Autoencoder] Epoch {epoch+1}/{epochs} - Reconstruction Loss: {epoch_loss:.6f}")
            
    train_time = time.time() - start_time
    return model, train_time

def find_ae_threshold(model, X_train_normal, percentile=97.0, device='cpu'):
    """Find threshold reconstruction error using normal training traffic."""
    model.eval()
    X_tensor = torch.tensor(X_train_normal, dtype=torch.float32).to(device)
    with torch.no_grad():
        reconstructed = model(X_tensor)
        # Compute mean squared error for each row
        errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1).cpu().numpy()
    
    threshold = np.percentile(errors, percentile)
    print(f"  [Autoencoder] Threshold determined at {percentile}th percentile: {threshold:.6f}")
    return threshold

def predict_autoencoder(model, X, threshold, device='cpu'):
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        reconstructed = model(X_tensor)
        errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1).cpu().numpy()
        # If reconstruction error > threshold, it's classified as an anomaly (1), else normal (0)
        preds = (errors > threshold).astype(int)
    # Return anomaly predictions and errors as proxy for anomaly score
    return preds, errors


# ==========================================
# 3. Traditional Machine Learning Wrappers
# ==========================================
def get_rf_model():
    return RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)

def get_xgb_model():
    return XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)

def get_svm_model():
    # Linear SVM using SGDClassifier for performance scaling on large datasets
    return SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, random_state=42, max_iter=1000, tol=1e-3)

def get_dt_model():
    return DecisionTreeClassifier(random_state=42)

def get_knn_model():
    return KNeighborsClassifier(n_neighbors=5, n_jobs=-1)

def get_nb_model():
    return GaussianNB()

def get_lr_model():
    return LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)

def get_lgbm_model():
    return LGBMClassifier(random_state=42, n_jobs=-1)

# ==========================================
# 4. PyTorch Long Short-Term Memory (LSTM)
# ==========================================
class LSTMClassifierNet(nn.Module):
    def __init__(self, input_dim):
        super(LSTMClassifierNet, self).__init__()
        self.hidden_dim = 64
        self.lstm = nn.LSTM(input_dim, self.hidden_dim, batch_first=True)
        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x is expected to be [batch, seq_len, input_dim]
        # For our tabular data, seq_len=1
        if x.dim() == 2:
            x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out

def train_lstm(X_train, y_train, epochs=8, batch_size=128, lr=0.001, device='cpu'):
    input_dim = X_train.shape[1]
    model = LSTMClassifierNet(input_dim).to(device)
    
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    start_time = time.time()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)
            
        epoch_loss /= len(X_train)
        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(f"  [LSTM] Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")
            
    train_time = time.time() - start_time
    return model, train_time

def predict_lstm(model, X, device='cpu'):
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        outputs = model(X_tensor)
        preds = (outputs >= 0.5).int().cpu().numpy().flatten()
        probs = outputs.cpu().numpy().flatten()
    return preds, probs

# ==========================================
# 5. PyTorch 1D Convolutional Neural Network (CNN)
# ==========================================
class CNN1DClassifierNet(nn.Module):
    def __init__(self, input_dim):
        super(CNN1DClassifierNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        # Calculate flattened dimension
        self.flattened_dim = 64 * (input_dim // 4) 
        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.flattened_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x is [batch, input_dim]. Conv1d expects [batch, channels, seq_len]
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = torch.flatten(x, 1)
        out = self.fc(x)
        return out

def train_cnn1d(X_train, y_train, epochs=8, batch_size=128, lr=0.001, device='cpu'):
    input_dim = X_train.shape[1]
    model = CNN1DClassifierNet(input_dim).to(device)
    
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    start_time = time.time()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)
            
        epoch_loss /= len(X_train)
        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(f"  [CNN1D] Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")
            
    train_time = time.time() - start_time
    return model, train_time

def predict_cnn1d(model, X, device='cpu'):
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        outputs = model(X_tensor)
        preds = (outputs >= 0.5).int().cpu().numpy().flatten()
        probs = outputs.cpu().numpy().flatten()
    return preds, probs

