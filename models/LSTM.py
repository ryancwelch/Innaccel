import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import os

class EHGDataset(Dataset):
    """Dataset class for EHG sequence data"""
    def __init__(self, X, y):
        """
        X: sequence data of shape (n_records, sequence_length, n_features)
        y: labels of shape (n_records, sequence_length)
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        """
        input_size: number of features per timestep
        hidden_size: number of LSTM units
        num_layers: number of LSTM layers
        dropout: dropout probability
        """
        super(BidirectionalLSTM, self).__init__()
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Classification layers
        self.fc1 = nn.Linear(hidden_size * 2, 64)  # *2 for bidirectional
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 1)  # Output 1 value per timestep
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # LSTM layer
        lstm_out, _ = self.lstm(x)  # shape: [batch_size, seq_len, hidden_size*2]
        
        # Apply classification layers to each timestep
        fc1_out = self.relu(self.fc1(lstm_out))
        fc1_out = self.dropout(fc1_out)
        output = self.sigmoid(self.fc2(fc1_out))  # shape: [batch_size, seq_len, 1]
        
        # Remove last dimension to match target shape
        output = output.squeeze(-1)  # shape: [batch_size, seq_len]
        return output

def train_model(model, train_loader, val_loader, epochs=50, learning_rate=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)
    
    # Create directory for saving model weights
    os.makedirs('model_weights', exist_ok=True)
    
    # Calculate class weights
    y_train = train_loader.dataset.y
    n_samples = y_train.numel()
    n_positive = y_train.sum().item()
    n_negative = n_samples - n_positive
    
    # Calculate weights inversely proportional to class frequencies
    weight_positive = n_samples / (2 * n_positive)
    weight_negative = n_samples / (2 * n_negative)
    
    # Create weight tensor and reshape to match target
    class_weights = torch.FloatTensor([weight_negative, weight_positive]).to(device)
    print(f"Class weights - Negative: {weight_negative:.2f}, Positive: {weight_positive:.2f}")
    
    criterion = nn.BCELoss(reduction='none')  # Don't reduce yet
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)  # shape: [batch_size, seq_len]
            
            # Calculate loss without reduction
            loss = criterion(outputs, batch_y)
            
            # Apply class weights manually
            weights = torch.where(batch_y == 1, 
                                torch.full_like(batch_y, weight_positive),
                                torch.full_like(batch_y, weight_negative))
            loss = (loss * weights).mean()  # Average over all elements
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            train_total += batch_y.numel()
            train_correct += (predicted == batch_y).sum().item()
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                
                # Calculate loss without reduction
                loss = criterion(outputs, batch_y)
                
                # Apply class weights manually
                weights = torch.where(batch_y == 1, 
                                    torch.full_like(batch_y, weight_positive),
                                    torch.full_like(batch_y, weight_negative))
                loss = (loss * weights).mean()  # Average over all elements
                
                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                val_total += batch_y.numel()
                val_correct += (predicted == batch_y).sum().item()
        
        # Calculate average losses and accuracies
        avg_train_loss = train_loss/len(train_loader)
        avg_val_loss = val_loss/len(val_loader)
        train_acc = 100*train_correct/train_total
        val_acc = 100*val_correct/val_total
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print('--------------------')
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'model_weights/best_lstm_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered!")
                break

def main():
    # Load data
    X = np.load('data/lstm_data/X_sequence.npy')
    y = np.load('data/lstm_data/y_sequence.npy')
    sequence_lengths = np.load('data/lstm_data/sequence_lengths.npy')
    
    print(f"Loaded data shapes - X: {X.shape}, y: {y.shape}")
    print(f"Sequence lengths range: {min(sequence_lengths)} - {max(sequence_lengths)}")
    print(f"Class distribution - Positive: {np.sum(y)}, Negative: {np.prod(y.shape) - np.sum(y)}")
    
    # Create record-level labels (whether each record contains any positive samples)
    record_labels = np.any(y, axis=1).astype(int)
    print(record_labels)
    
    # Split data at record level
    indices = np.arange(X.shape[0])
    print(indices.shape)
    train_idx, val_idx = train_test_split(
        indices, 
        test_size=0.5, 
        random_state=42, 
        stratify=record_labels
    )
    
    # Split the data
    X_train = X[train_idx]
    X_val = X[val_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]
    
    print(f"\nSplit data shapes:")
    print(f"Train - X: {X_train.shape}, y: {y_train.shape}")
    print(f"Val - X: {X_val.shape}, y: {y_val.shape}")
    
    # Create datasets
    train_dataset = EHGDataset(X_train, y_train)
    val_dataset = EHGDataset(X_val, y_val)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=32, 
        shuffle=False,
        num_workers=4
    )
    
    # Initialize model
    input_size = X.shape[2]  # number of features
    model = BidirectionalLSTM(
        input_size=input_size,
        hidden_size=128,
        num_layers=2,
        dropout=0.3
    )
    
    # Print model summary
    print("\nModel Architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Train model
    train_model(
        model,
        train_loader,
        val_loader,
        epochs=50,
        learning_rate=0.001
    )

if __name__ == "__main__":
    main()