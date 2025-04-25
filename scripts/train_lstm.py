#!/usr/bin/env python
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import pickle
import time

sys.path.append(os.path.abspath('.'))
from models.LSTM import BidirectionalLSTM, EHGDataset, train_model

def load_contraction_dataset(data_dir="data/contraction_data"):
    """
    Load the prepared contraction detection dataset.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing the dataset
        
    Returns:
    --------
    X : np.ndarray
        Features
    y : np.ndarray
        Labels
    records : np.ndarray
        Record names for each window
    feature_names : list
        List of feature names
    dataset_info : dict
        Information about the dataset
    """
    try:
        # Load full dataset
        X = np.load(os.path.join(data_dir, 'X.npy'))
        y = np.load(os.path.join(data_dir, 'y.npy'))
        records = np.load(os.path.join(data_dir, 'records.npy'))
        feature_names = np.load(os.path.join(data_dir, 'feature_names.npy'), allow_pickle=True)
        dataset_info = np.load(os.path.join(data_dir, 'dataset_info.npy'), allow_pickle=True).item()
        
        print(f"Loaded dataset from {data_dir}")
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        print(f"Number of unique records: {len(np.unique(records))}")
        
        return X, y, records, feature_names, dataset_info
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None, None, None, None

def prepare_full_recording_sequences(X, y, records, min_windows=20, max_pad=None):
    """
    Convert windowed data to sequence format required by LSTM,
    treating each full recording as a single sequence.
    
    Parameters:
    -----------
    X : np.ndarray
        Features array of shape (n_windows, n_features)
    y : np.ndarray
        Labels array of shape (n_windows,)
    records : np.ndarray
        Record names for each window
    min_windows : int
        Minimum number of windows required to include a recording
    max_pad : int or None
        Maximum sequence length for padding (if None, use the longest recording)
    
    Returns:
    --------
    X_sequence : np.ndarray
        Sequence data of shape (n_records, max_seq_length, n_features)
    y_sequence : np.ndarray
        Labels for each timestep in sequence (n_records, max_seq_length)
    record_names : np.ndarray
        Record names for each sequence
    sequence_lengths : np.ndarray
        Actual length of each sequence before padding
    """
    print("Preparing full recording sequences...")
    
    # Get unique records
    unique_records = np.unique(records)
    print(f"Processing {len(unique_records)} unique records")
    
    # First pass: get sequence lengths for each record
    record_lengths = {}
    valid_records = []
    
    for record in unique_records:
        record_mask = (records == record)
        length = np.sum(record_mask)
        
        if length >= min_windows:
            record_lengths[record] = length
            valid_records.append(record)
        else:
            print(f"  Skipping record {record} with only {length} windows (< {min_windows})")
    
    if not valid_records:
        raise ValueError("No valid records found with minimum window requirements")
    
    # Determine max sequence length for padding
    if max_pad is None:
        max_seq_length = max(record_lengths.values())
    else:
        max_seq_length = min(max(record_lengths.values()), max_pad)
    
    print(f"Max sequence length: {max_seq_length} windows")
    
    # Prepare arrays
    n_features = X.shape[1]
    X_sequence = np.zeros((len(valid_records), max_seq_length, n_features))
    y_sequence = np.zeros((len(valid_records), max_seq_length))
    record_names = np.array(valid_records)
    sequence_lengths = np.zeros(len(valid_records), dtype=int)
    
    # Second pass: create sequences
    for i, record in enumerate(tqdm(valid_records)):
        record_mask = (records == record)
        record_X = X[record_mask]
        record_y = y[record_mask]
        
        # Store actual sequence length
        seq_length = min(len(record_X), max_seq_length)
        sequence_lengths[i] = seq_length
        
        # Add data to arrays (with truncation if needed)
        X_sequence[i, :seq_length, :] = record_X[:seq_length]
        y_sequence[i, :seq_length] = record_y[:seq_length]
    
    print(f"Created {len(valid_records)} sequences")
    print(f"X_sequence shape: {X_sequence.shape}")
    print(f"y_sequence shape: {y_sequence.shape}")
    print(f"Average sequence length: {np.mean(sequence_lengths):.2f} windows")
    
    return X_sequence, y_sequence, record_names, sequence_lengths

def normalize_sequence_data(X_sequence, sequence_lengths=None):
    """
    Normalize features across the sequence dimension.
    
    Parameters:
    -----------
    X_sequence : np.ndarray
        Sequence data of shape (n_sequences, seq_length, n_features)
    sequence_lengths : np.ndarray or None
        Actual length of each sequence before padding
    
    Returns:
    --------
    X_normalized : np.ndarray
        Normalized sequence data
    normalization_params : dict
        Parameters used for normalization
    """
    # Create a mask for actual data (not padding)
    n_seq, seq_len, n_features = X_sequence.shape
    
    if sequence_lengths is not None:
        # Create a mask to exclude padding
        mask = np.zeros((n_seq, seq_len), dtype=bool)
        for i, length in enumerate(sequence_lengths):
            mask[i, :length] = True
        
        # Reshape to combine sequence and batch dimensions for actual data only
        valid_data = X_sequence[mask].reshape(-1, n_features)
    else:
        valid_data = X_sequence.reshape(-1, n_features)
    
    # Calculate mean and std on valid data only
    mean = np.nanmean(valid_data, axis=0)
    std = np.nanstd(valid_data, axis=0)
    
    # Replace zeros in std to avoid division by zero
    std[std == 0] = 1.0
    
    # Replace NaN values
    X_copy = np.copy(X_sequence)
    X_copy = np.nan_to_num(X_copy)
    
    # Normalize all data (including padding)
    X_normalized = (X_copy - mean) / std
    
    # Store parameters
    normalization_params = {
        'mean': mean,
        'std': std
    }
    
    return X_normalized, normalization_params

class PaddedSequenceDataset(torch.utils.data.Dataset):
    """Dataset for padded variable-length sequences"""
    def __init__(self, X, y, sequence_lengths=None):
        """
        Parameters:
        -----------
        X : np.ndarray
            Sequence data of shape (n_records, max_seq_length, n_features)
        y : np.ndarray
            Labels of shape (n_records, max_seq_length)
        sequence_lengths : np.ndarray or None
            Actual length of each sequence before padding
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.sequence_lengths = sequence_lengths
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.sequence_lengths is not None:
            # Return sequence data, labels, and length
            return self.X[idx], self.y[idx], self.sequence_lengths[idx]
        else:
            return self.X[idx], self.y[idx]

class CustomBidirectionalLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        """
        Modified LSTM model with masking for padded sequences.

        Parameters:
        -----------
        input_size : int
            Number of features per timestep
        hidden_size : int
            Number of LSTM units
        num_layers : int
            Number of LSTM layers
        dropout : float
            Dropout probability
        """
        super(CustomBidirectionalLSTM, self).__init__()
        
        # LSTM layers
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Classification layers
        self.fc1 = torch.nn.Linear(hidden_size * 2, 64)  # *2 for bidirectional
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)
        self.fc2 = torch.nn.Linear(64, 1)  # Output 1 value per timestep
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, x, lengths=None):
        # Pack the sequences to handle variable length
        if lengths is not None:
            # Sort sequences by length in descending order
            lengths_np = lengths.cpu().numpy() if isinstance(lengths, torch.Tensor) else lengths
            sorted_lengths, sorted_idx = torch.sort(torch.tensor(lengths_np), descending=True)
            sorted_x = x[sorted_idx]
            
            # Pack sequences
            packed_input = torch.nn.utils.rnn.pack_padded_sequence(
                sorted_x, sorted_lengths.cpu().numpy(), batch_first=True
            )
            
            # Run through LSTM
            packed_output, _ = self.lstm(packed_input)
            
            # Unpack the output
            output, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
            
            # Restore original order
            _, original_idx = torch.sort(sorted_idx)
            output = output[original_idx]
        else:
            # Regular forward pass without packing
            output, _ = self.lstm(x)  # shape: [batch_size, seq_len, hidden_size*2]
        
        # Apply classification layers to each timestep
        fc1_out = self.relu(self.fc1(output))
        fc1_out = self.dropout(fc1_out)
        output = self.sigmoid(self.fc2(fc1_out))  # shape: [batch_size, seq_len, 1]
        
        # Remove last dimension to match target shape
        output = output.squeeze(-1)  # shape: [batch_size, seq_len]
        return output

def custom_train_model(model, train_loader, val_loader, sequence_lengths=True, 
                      epochs=50, learning_rate=0.001, output_dir="model_weights"):
    """
    Train model with support for variable-length sequences.
    
    Parameters:
    -----------
    model : torch.nn.Module
        Model to train
    train_loader : DataLoader
        DataLoader for training data
    val_loader : DataLoader
        DataLoader for validation data
    sequence_lengths : bool
        Whether the DataLoader includes sequence lengths
    epochs : int
        Number of training epochs
    learning_rate : float
        Learning rate for optimizer
    output_dir : str
        Directory to save model weights
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)
    
    # Create directory for saving model weights
    os.makedirs(output_dir, exist_ok=True)
    
    # Create weight tensor
    # We'll calculate this from the data
    pos_count = 0
    neg_count = 0
    total_elements = 0
    
    for batch in train_loader:
        if sequence_lengths:
            batch_X, batch_y, batch_lengths = batch
            for i, length in enumerate(batch_lengths):
                length = length.item() if isinstance(length, torch.Tensor) else length
                pos_count += torch.sum(batch_y[i, :length] == 1).item()
                neg_count += torch.sum(batch_y[i, :length] == 0).item()
                total_elements += length
        else:
            batch_X, batch_y = batch
            pos_count += torch.sum(batch_y == 1).item()
            neg_count += torch.sum(batch_y == 0).item()
            total_elements = batch_y.numel()
    
    # Calculate weights inversely proportional to class frequencies
    weight_positive = total_elements / (2 * pos_count) if pos_count > 0 else 1.0
    weight_negative = total_elements / (2 * neg_count) if neg_count > 0 else 1.0
    
    print(f"Class weights - Negative: {weight_negative:.2f}, Positive: {weight_positive:.2f}")
    
    # Loss function and optimizer
    criterion = torch.nn.BCELoss(reduction='none')  # Don't reduce yet
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
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
        
        for batch_idx, batch in enumerate(train_loader):
            if sequence_lengths:
                batch_X, batch_y, batch_lengths = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
                outputs = model(batch_X, batch_lengths)
                
                # Create mask for actual sequence data (not padding)
                mask = torch.zeros_like(batch_y, dtype=torch.bool)
                for i, length in enumerate(batch_lengths):
                    length = length.item() if isinstance(length, torch.Tensor) else length
                    mask[i, :length] = True
                
                # Calculate loss only on actual sequence data
                loss = criterion(outputs, batch_y)
                loss = loss * mask.float()  # Zero out loss for padding
                
                # Apply class weights manually
                weights = torch.where(batch_y == 1, 
                                    torch.full_like(batch_y, weight_positive),
                                    torch.full_like(batch_y, weight_negative))
                weights = weights * mask.float()  # Zero out weights for padding
                
                loss = (loss * weights).sum() / weights.sum()  # Weighted average
                
                # Count correct predictions (only for actual sequence data)
                predicted = (outputs > 0.5).float()
                train_total += mask.sum().item()
                train_correct += ((predicted == batch_y) * mask).sum().item()
                
            else:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                
                # Calculate loss without reduction
                loss = criterion(outputs, batch_y)
                
                # Apply class weights manually
                weights = torch.where(batch_y == 1, 
                                    torch.full_like(batch_y, weight_positive),
                                    torch.full_like(batch_y, weight_negative))
                loss = (loss * weights).mean()  # Average over all elements
                
                # Count correct predictions
                predicted = (outputs > 0.5).float()
                train_total += batch_y.numel()
                train_correct += (predicted == batch_y).sum().item()
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
            # Print progress every 5 batches
            if (batch_idx + 1) % 5 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}, Acc: {100*train_correct/max(1, train_total):.2f}%')
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if sequence_lengths:
                    batch_X, batch_y, batch_lengths = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
                    outputs = model(batch_X, batch_lengths)
                    
                    # Create mask for actual sequence data (not padding)
                    mask = torch.zeros_like(batch_y, dtype=torch.bool)
                    for i, length in enumerate(batch_lengths):
                        length = length.item() if isinstance(length, torch.Tensor) else length
                        mask[i, :length] = True
                    
                    # Calculate loss only on actual sequence data
                    loss = criterion(outputs, batch_y)
                    loss = loss * mask.float()  # Zero out loss for padding
                    
                    # Apply class weights manually
                    weights = torch.where(batch_y == 1, 
                                        torch.full_like(batch_y, weight_positive),
                                        torch.full_like(batch_y, weight_negative))
                    weights = weights * mask.float()  # Zero out weights for padding
                    
                    loss = (loss * weights).sum() / weights.sum()  # Weighted average
                    
                    # Count correct predictions (only for actual sequence data)
                    predicted = (outputs > 0.5).float()
                    val_total += mask.sum().item()
                    val_correct += ((predicted == batch_y) * mask).sum().item()
                    
                else:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    
                    # Calculate loss without reduction
                    loss = criterion(outputs, batch_y)
                    
                    # Apply class weights manually
                    weights = torch.where(batch_y == 1, 
                                        torch.full_like(batch_y, weight_positive),
                                        torch.full_like(batch_y, weight_negative))
                    loss = (loss * weights).mean()  # Average over all elements
                    
                    # Count correct predictions
                    predicted = (outputs > 0.5).float()
                    val_total += batch_y.numel()
                    val_correct += (predicted == batch_y).sum().item()
                
                val_loss += loss.item()
        
        # Calculate average losses and accuracies
        avg_train_loss = train_loss/len(train_loader)
        avg_val_loss = val_loss/len(val_loader)
        train_acc = 100*train_correct/max(1, train_total)
        val_acc = 100*val_correct/max(1, val_total)
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print('--------------------')
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_lstm_model.pth'))
            print(f"New best model saved with validation loss: {best_val_loss:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs!")
                break
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(output_dir, 'final_lstm_model.pth'))
    print(f"Final model saved after {epoch+1} epochs.")

def save_sequence_data(X_sequence, y_sequence, sequence_lengths, normalization_params, output_dir):
    """
    Save prepared sequence data to disk.
    
    Parameters:
    -----------
    X_sequence : np.ndarray
        Sequence data
    y_sequence : np.ndarray
        Sequence labels
    sequence_lengths : np.ndarray
        Length of each sequence
    normalization_params : dict
        Parameters used for normalization
    output_dir : str
        Directory to save data
    """
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, 'X_sequence.npy'), X_sequence)
    np.save(os.path.join(output_dir, 'y_sequence.npy'), y_sequence)
    np.save(os.path.join(output_dir, 'sequence_lengths.npy'), sequence_lengths)
    
    with open(os.path.join(output_dir, 'normalization_params.pkl'), 'wb') as f:
        pickle.dump(normalization_params, f)
    
    print(f"Saved sequence data to {output_dir}")

def train_lstm_model(X_sequence, y_sequence, sequence_lengths, output_dir, batch_size=8, epochs=50):
    """
    Train LSTM model on prepared sequence data.
    
    Parameters:
    -----------
    X_sequence : np.ndarray
        Sequence data of shape (n_sequences, seq_length, n_features)
    y_sequence : np.ndarray
        Labels for each timestep in sequence (n_sequences, seq_length)
    sequence_lengths : np.ndarray
        Length of each sequence
    output_dir : str
        Directory to save model weights and logs
    batch_size : int
        Batch size for training
    epochs : int
        Number of training epochs
    """
    # Create record-level labels (whether each record contains any positive samples)
    record_labels = np.zeros(len(X_sequence), dtype=int)
    for i, length in enumerate(sequence_lengths):
        record_labels[i] = 1 if np.any(y_sequence[i, :length]) else 0
    
    print(f"Record-level class distribution: {np.bincount(record_labels, minlength=2)}")
    
    # Split data at record level
    from sklearn.model_selection import train_test_split
    
    indices = np.arange(X_sequence.shape[0])
    
    # Check if stratification is possible (need at least 2 examples per class)
    min_class_count = np.min(np.bincount(record_labels, minlength=2))
    
    if min_class_count < 2:
        print(f"Warning: Insufficient examples in class {np.argmin(np.bincount(record_labels, minlength=2))} for stratification (only {min_class_count}).")
        print("Performing random split without stratification.")
        train_idx, val_idx = train_test_split(
            indices, 
            test_size=0.2, 
            random_state=42
        )
    else:
        print("Performing stratified split to maintain class balance.")
        train_idx, val_idx = train_test_split(
            indices, 
            test_size=0.2, 
            random_state=42, 
            stratify=record_labels
        )
    
    # Split the data
    X_train = X_sequence[train_idx]
    X_val = X_sequence[val_idx]
    y_train = y_sequence[train_idx]
    y_val = y_sequence[val_idx]
    train_lengths = sequence_lengths[train_idx]
    val_lengths = sequence_lengths[val_idx]
    
    # Print split statistics
    train_labels = np.zeros(len(train_idx), dtype=int)
    val_labels = np.zeros(len(val_idx), dtype=int)
    for i, idx in enumerate(train_idx):
        train_labels[i] = record_labels[idx]
    for i, idx in enumerate(val_idx):
        val_labels[i] = record_labels[idx]
    
    print(f"Train set class distribution: {np.bincount(train_labels, minlength=2)}")
    print(f"Validation set class distribution: {np.bincount(val_labels, minlength=2)}")
    
    print(f"\nSplit data shapes:")
    print(f"Train - X: {X_train.shape}, y: {y_train.shape}, records: {len(train_idx)}")
    print(f"Val - X: {X_val.shape}, y: {y_val.shape}, records: {len(val_idx)}")
    
    # Create datasets
    train_dataset = PaddedSequenceDataset(X_train, y_train, train_lengths)
    val_dataset = PaddedSequenceDataset(X_val, y_val, val_lengths)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4
    )
    
    # Initialize model
    input_size = X_sequence.shape[2]  # number of features
    model = CustomBidirectionalLSTM(
        input_size=input_size,
        hidden_size=128,
        num_layers=2,
        dropout=0.3
    )
    
    # Print model summary
    print("\nModel Architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Create model output directory
    model_dir = os.path.join(output_dir, 'model_weights')
    os.makedirs(model_dir, exist_ok=True)
    
    # Train model
    start_time = time.time()
    custom_train_model(
        model,
        train_loader,
        val_loader,
        sequence_lengths=True,
        epochs=epochs,
        learning_rate=0.001,
        output_dir=model_dir
    )
    training_time = time.time() - start_time
    print(f"Training completed in {training_time/60:.2f} minutes")

def main():
    parser = argparse.ArgumentParser(
        description='Train LSTM model for contraction detection using full-recording sequences',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--data_dir', type=str, default='data/contraction_data',
                      help='Directory containing the prepared contraction dataset')
    parser.add_argument('--output_dir', type=str, default='data/lstm_data',
                      help='Directory to save sequence data and model')
    parser.add_argument('--min_windows', type=int, default=50,
                      help='Minimum number of windows required to include a recording')
    parser.add_argument('--max_length', type=int, default=None,
                      help='Maximum sequence length (None to use the longest recording)')
    parser.add_argument('--batch_size', type=int, default=4,
                      help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                      help='Number of training epochs')
    parser.add_argument('--skip_preparation', action='store_true',
                      help='Skip data preparation and load existing sequence data')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if not args.skip_preparation:
        # Load dataset
        X, y, records, feature_names, dataset_info = load_contraction_dataset(args.data_dir)
        
        if X is None:
            print("Failed to load dataset. Exiting.")
            sys.exit(1)
        
        # Prepare sequence data (one sequence per recording)
        X_sequence, y_sequence, record_names, sequence_lengths = prepare_full_recording_sequences(
            X, y, records,
            min_windows=args.min_windows,
            max_pad=args.max_length
        )
        
        # Normalize sequence data
        X_sequence, normalization_params = normalize_sequence_data(X_sequence, sequence_lengths)
        
        # Save sequence data
        save_sequence_data(
            X_sequence, y_sequence, sequence_lengths, 
            normalization_params, args.output_dir
        )
    else:
        print("Loading prepared sequence data...")
        try:
            X_sequence = np.load(os.path.join(args.output_dir, 'X_sequence.npy'))
            y_sequence = np.load(os.path.join(args.output_dir, 'y_sequence.npy'))
            sequence_lengths = np.load(os.path.join(args.output_dir, 'sequence_lengths.npy'))
            print(f"Loaded sequence data - X: {X_sequence.shape}, y: {y_sequence.shape}")
            print(f"Number of sequences: {len(sequence_lengths)}")
            print(f"Average sequence length: {np.mean(sequence_lengths):.2f} windows")
        except Exception as e:
            print(f"Error loading sequence data: {e}")
            sys.exit(1)
    
    # Train LSTM model
    train_lstm_model(
        X_sequence, 
        y_sequence,
        sequence_lengths,
        args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs
    )
    
    print("Training complete!")

if __name__ == "__main__":
    main()
