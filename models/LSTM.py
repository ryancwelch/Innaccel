import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class EHGDataset(Dataset):
    """Dataset class for EHG data"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super(BidirectionalLSTM, self).__init__()
        
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        self.dropout = nn.Dropout(dropout)
        
        self.lstm2 = nn.LSTM(
            input_size=hidden_size*2,  # *2 because bidirectional
            hidden_size=hidden_size//2,  # //2 to reduce dimensionality
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # First LSTM layer
        lstm1_out, _ = self.lstm1(x)
        lstm1_out = self.dropout(lstm1_out)
        
        # Second LSTM layer
        lstm2_out, _ = self.lstm2(lstm1_out)
        lstm2_out = lstm2_out[:, -1, :]  # Take last timestep
        
        # Fully connected layers
        fc1_out = self.relu(self.fc1(lstm2_out))
        fc2_out = self.sigmoid(self.fc2(fc1_out))
        
        return fc2_out

def train_model(model, train_loader, val_loader, epochs=10, learning_rate=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y.unsqueeze(1))
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            train_total += batch_y.size(0)
            train_correct += (predicted.squeeze() == batch_y).sum().item()
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y.unsqueeze(1))
                
                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                val_total += batch_y.size(0)
                val_correct += (predicted.squeeze() == batch_y).sum().item()
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}, '
              f'Train Acc: {100*train_correct/train_total:.2f}%')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}, '
              f'Val Acc: {100*val_correct/val_total:.2f}%')
        print('--------------------')

def main():
    # Load data
    X = np.load('data/lstm_data/X_sequence.npy')
    y = np.load('data/lstm_data/y_sequence.npy')

        # Reshape for LSTM [batch, sequence_length, features]
    X = X.reshape(-1, 1, X.shape[1])

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create datasets
    train_dataset = EHGDataset(X_train, y_train)
    val_dataset = EHGDataset(X_val, y_val)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=32, 
        shuffle=False
    )
    
    # Initialize model
    input_size = X.shape[2]  # number of features
    model = BidirectionalLSTM(input_size=input_size)

    # Train model
    train_model(model, train_loader, val_loader, epochs=10)
    
    # Save model
    torch.save(model.state_dict(), 'models/lstm_model.pth')

if __name__ == "__main__":
    main()