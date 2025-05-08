#!/usr/bin/env python
"""
Train a CNN-LSTM hybrid model on sequential contraction data.
Combines CNN for feature extraction with LSTM for temporal dependencies.
Handles variable-length sequences with proper masking.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import time

class SequenceDataset(Dataset):
    """Dataset for variable-length sequence data with masking for padded values"""
    def __init__(self, X, y, masks=None):
        """
        Args:
            X: numpy array of shape [num_patients, num_windows, num_features] (no NaNs)
            y: numpy array of shape [num_patients, num_windows] (no NaNs)
            masks: numpy array of shape [num_patients, num_windows] (True for valid values)
        """
        # Convert inputs to tensors
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
        # Create or use masks for valid windows
        if masks is not None:
            self.masks = torch.BoolTensor(masks)
        else:
            # Default mask (all True) if not provided
            self.masks = torch.ones_like(self.y, dtype=torch.bool)
        
        # Get sequence lengths (number of valid windows per patient)
        self.seq_lengths = torch.sum(self.masks, dim=1).int()
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.masks[idx], self.seq_lengths[idx]

class CNN_LSTM(nn.Module):
    """CNN-LSTM hybrid for sequence classification"""
    def __init__(self, input_size, hidden_size=128, num_layers=2, 
                 cnn_filters=[64, 128], kernel_sizes=[3, 5], dropout=0.3):
        super(CNN_LSTM, self).__init__()
        
        # CNN layers for feature extraction
        self.cnn_layers = nn.ModuleList()
        
        # First CNN layer takes raw input
        self.cnn_layers.append(
            nn.Sequential(
                nn.Conv1d(input_size, cnn_filters[0], kernel_size=kernel_sizes[0], padding=kernel_sizes[0]//2),
                nn.BatchNorm1d(cnn_filters[0]),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout/2)
            )
        )
        
        # Add additional CNN layers
        for i in range(1, len(cnn_filters)):
            self.cnn_layers.append(
                nn.Sequential(
                    nn.Conv1d(cnn_filters[i-1], cnn_filters[i], kernel_size=kernel_sizes[i], padding=kernel_sizes[i]//2),
                    nn.BatchNorm1d(cnn_filters[i]),
                    nn.LeakyReLU(0.1),
                    nn.Dropout(dropout/2)
                )
            )
        
        # Add a residual connection
        self.has_residual = (len(cnn_filters) > 1)
        if self.has_residual:
            self.residual_conv = nn.Conv1d(input_size, cnn_filters[-1], kernel_size=1)
            self.residual_bn = nn.BatchNorm1d(cnn_filters[-1])
        
        # LSTM layers to process CNN features
        self.lstm = nn.LSTM(
            input_size=cnn_filters[-1],
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Classification layers
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(64, 1)  # Output 1 value per timestep
        
    def forward(self, x, seq_lengths=None):
        batch_size, seq_len, n_features = x.size()
        orig_seq_len = seq_len  # Store original sequence length
        
        # Reshape for CNN: [batch_size, seq_len, n_features] -> [batch_size, n_features, seq_len]
        x = x.permute(0, 2, 1)
        
        # Save input for residual connection
        residual = x if self.has_residual else None
        
        # Apply CNN layers for feature extraction
        for cnn_layer in self.cnn_layers:
            x = cnn_layer(x)
        
        # Apply residual connection
        if self.has_residual:
            res = self.residual_conv(residual)
            res = self.residual_bn(res)
            x = x + res
        
        # Reshape back for LSTM: [batch_size, n_cnn_filters, seq_len] -> [batch_size, seq_len, n_cnn_filters]
        x = x.permute(0, 2, 1)
        
        # Get the current sequence length after CNN processing
        curr_seq_len = x.size(1)
        
        # If the sequence length has changed, resize it back to the original length
        if curr_seq_len != orig_seq_len:
            # Use adaptive pooling to match the original sequence length
            adaptive_pool = nn.AdaptiveAvgPool1d(orig_seq_len)
            x = x.permute(0, 2, 1)  # [B, hidden, seq_len]
            x = adaptive_pool(x)
            x = x.permute(0, 2, 1)  # [B, seq_len, hidden]
        
        # LSTM layer - process without packing since we'll handle masking in the loss function
        lstm_out, _ = self.lstm(x)  # shape: [batch_size, seq_len, hidden_size*2]
        
        # Apply attention mechanism
        attn_weights = self.attention(lstm_out)  # [batch_size, seq_len, 1]
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # Keep original lstm_out for classification but also compute attention-weighted output
        context_vector = torch.bmm(attn_weights.transpose(1, 2), lstm_out)  # [batch_size, 1, hidden_size*2]
        context_vector = context_vector.squeeze(1)  # [batch_size, hidden_size*2]
        
        # Prepare for full sequence output
        # Apply classification layers
        # First reshape lstm_out for batch norm: [batch_size*seq_len, hidden_size*2]
        batch_seq = lstm_out.size(0) * lstm_out.size(1)
        hidden_size = lstm_out.size(2)
        
        lstm_out_flat = lstm_out.reshape(-1, hidden_size)
        fc1_out = F.leaky_relu(self.fc1(lstm_out_flat), 0.1)
        fc1_out = fc1_out.reshape(batch_size, seq_len, -1)
        
        # Apply classification layers
        output = F.leaky_relu(self.fc1(lstm_out), 0.1)
        output = self.dropout(output)
        output = F.leaky_relu(self.fc2(output), 0.1)
        output = self.dropout(output)
        output = torch.sigmoid(self.output(output)).squeeze(-1)  # shape: [batch_size, seq_len]
        
        return output

def train_model(model, train_loader, val_loader, device, epochs=30, 
                learning_rate=0.001, output_dir='model_output'):
    """Train the model and save results"""
    os.makedirs(output_dir, exist_ok=True)
    model_dir = os.path.join(output_dir, 'model')
    os.makedirs(model_dir, exist_ok=True)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Track metrics
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    no_improve_count = 0
    
    # Calculate class weights based on entire training set once
    print("Computing dataset class balance for proper weighting...")
    all_y = []
    all_masks = []
    for _, y_batch, mask_batch, _ in train_loader:
        valid_y = y_batch[mask_batch.bool()]
        all_y.extend(valid_y.flatten().cpu().numpy())
        all_masks.extend(mask_batch.flatten().cpu().numpy())
    
    all_y = np.array(all_y)
    pos_count = np.sum(all_y == 1)
    neg_count = np.sum(all_y == 0)
    total = pos_count + neg_count
    
    if total > 0 and pos_count > 0 and neg_count > 0:
        # Set higher weight for positive class (contractions) due to imbalance
        pos_weight = total / (pos_count * 2)  # Extra weight on positives
        neg_weight = total / (neg_count * 2)
        print(f"Class weights - Positive: {pos_weight:.4f}, Negative: {neg_weight:.4f}")
        print(f"Class balance - Positive: {pos_count}/{total} ({pos_count/total:.2%}), Negative: {neg_count}/{total} ({neg_count/total:.2%})")
    else:
        pos_weight = 5.0  # Default to high positive weight if calculation fails
        neg_weight = 1.0
        print("Warning: Could not calculate class weights, using defaults")
    
    # Training loop
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_samples = 0
        
        for X_batch, y_batch, mask_batch, len_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            mask_batch = mask_batch.to(device)
            len_batch = len_batch.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            y_pred = model(X_batch, len_batch)
            
            # Calculate weighted binary cross entropy with focal loss component
            # Only consider non-padded elements
            y_ones = y_batch == 1
            y_zeros = (y_batch == 0) & mask_batch  # Only masked zeros
            
            # Focal loss parameters
            gamma = 2.0  # Focusing parameter (2.0 is common)
            
            # Calculate focal loss
            # pt = p for y=1, pt = 1-p for y=0
            pt_pos = y_pred
            pt_neg = 1 - y_pred
            
            # (1-pt)^gamma * log(pt)
            focal_pos = torch.pow(1 - pt_pos, gamma) * torch.log(pt_pos + 1e-7)
            focal_neg = torch.pow(1 - pt_neg, gamma) * torch.log(pt_neg + 1e-7)
            
            # Combine with class weights
            bce_pos = -focal_pos * y_batch * pos_weight
            bce_neg = -focal_neg * (1 - y_batch) * neg_weight
            
            # Sum positive and negative losses
            weighted_bce = bce_pos + bce_neg
            
            # Apply masking
            masked_bce = weighted_bce * mask_batch
            
            # Average loss - only over non-padded elements
            loss = torch.sum(masked_bce) / torch.sum(mask_batch)
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer.step()
            
            # Track metrics
            train_loss += loss.item() * torch.sum(mask_batch).item()
            train_samples += torch.sum(mask_batch).item()
        
        # Calculate epoch loss
        epoch_train_loss = train_loss / train_samples if train_samples > 0 else 0
        train_losses.append(epoch_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        val_samples = 0
        
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for X_batch, y_batch, mask_batch, len_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                mask_batch = mask_batch.to(device)
                len_batch = len_batch.to(device)
                
                # Forward pass
                y_pred = model(X_batch, len_batch)
                
                # Collect predictions and labels for metrics
                for i in range(len(X_batch)):
                    valid_mask = mask_batch[i].bool()
                    if valid_mask.sum() > 0:
                        val_labels.extend(y_batch[i][valid_mask].cpu().numpy())
                        val_preds.extend((y_pred[i][valid_mask] > 0.5).int().cpu().numpy())
                
                # Calculate loss (same as training)
                y_ones = y_batch == 1
                y_zeros = (y_batch == 0) & mask_batch
                
                # Focal loss parameters
                gamma = 2.0
                
                # Calculate focal loss
                pt_pos = y_pred
                pt_neg = 1 - y_pred
                
                focal_pos = torch.pow(1 - pt_pos, gamma) * torch.log(pt_pos + 1e-7)
                focal_neg = torch.pow(1 - pt_neg, gamma) * torch.log(pt_neg + 1e-7)
                
                # Combine with class weights
                bce_pos = -focal_pos * y_batch * pos_weight
                bce_neg = -focal_neg * (1 - y_batch) * neg_weight
                
                weighted_bce = bce_pos + bce_neg
                masked_bce = weighted_bce * mask_batch
                
                batch_loss = torch.sum(masked_bce) / torch.sum(mask_batch) if torch.sum(mask_batch) > 0 else 0
                
                val_loss += batch_loss.item() * torch.sum(mask_batch).item()
                val_samples += torch.sum(mask_batch).item()
        
        # Calculate epoch validation loss
        epoch_val_loss = val_loss / val_samples if val_samples > 0 else 0
        val_losses.append(epoch_val_loss)
        
        # Calculate validation metrics for reporting
        val_labels = np.array(val_labels)
        val_preds = np.array(val_preds)
        
        # Compute accuracy and F1 score directly
        if len(val_labels) > 0 and len(np.unique(val_labels)) > 1:
            val_accuracy = np.mean(val_labels == val_preds)
            
            # Compute F1 score
            true_pos = np.sum((val_labels == 1) & (val_preds == 1))
            false_pos = np.sum((val_labels == 0) & (val_preds == 1))
            false_neg = np.sum((val_labels == 1) & (val_preds == 0))
            
            precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
            recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
            
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        else:
            val_accuracy = 0
            f1 = 0
        
        # Print progress
        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}')
        print(f'  Val Metrics - Acc: {val_accuracy:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')
        
        # Save history at each epoch for real-time monitoring
        with open(os.path.join(output_dir, 'training_progress.txt'), 'a') as f:
            f.write(f'Epoch {epoch+1}: Train Loss = {epoch_train_loss:.6f}, Val Loss = {epoch_val_loss:.6f}\n')
            f.write(f'  Val Metrics - Acc: {val_accuracy:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}\n')
        
        # Update learning rate
        scheduler.step(epoch_val_loss)
        
        # Save best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), os.path.join(model_dir, 'best_model.pt'))
            print(f'Saved new best model with validation loss: {best_val_loss:.4f}')
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= 5:  # Early stopping
                print('Early stopping triggered')
                break
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(model_dir, 'final_model.pt'))
    
    # Save training history
    np.save(os.path.join(output_dir, 'train_losses.npy'), np.array(train_losses))
    np.save(os.path.join(output_dir, 'val_losses.npy'), np.array(val_losses))
    
    # Plot loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'loss_curves.png'))
    
    # Load best model for evaluation
    model.load_state_dict(torch.load(os.path.join(model_dir, 'best_model.pt')))
    return model

def evaluate_model(model, test_loader, device, output_dir):
    """Evaluate the model and save metrics"""
    os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for X_batch, y_batch, mask_batch, len_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            mask_batch = mask_batch.to(device)
            len_batch = len_batch.to(device)
            
            # Get predictions
            y_pred = model(X_batch, len_batch)
            
            # Collect only valid predictions (non-padded)
            for i in range(len(X_batch)):
                valid_mask = mask_batch[i].bool()
                if valid_mask.sum() > 0:
                    all_labels.extend(y_batch[i][valid_mask].cpu().numpy())
                    all_probs.extend(y_pred[i][valid_mask].cpu().numpy())
                    all_preds.extend((y_pred[i][valid_mask] > 0.5).int().cpu().numpy())
    
    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    report = classification_report(all_labels, all_preds, output_dict=True)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    # Calculate AUC
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    # Calculate additional metrics
    TN = conf_matrix[0, 0]
    FP = conf_matrix[0, 1]
    FN = conf_matrix[1, 0]
    TP = conf_matrix[1, 1]
    
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    ppv = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    npv = TN / (TN + FN) if (TN + FN) > 0 else 0.0
    f1 = report['1']['f1-score'] if '1' in report else 0.0
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0
    
    # Save results
    with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
        f.write("CNN-LSTM Model Evaluation Results\n")
        f.write("================================\n\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"ROC AUC: {roc_auc:.4f}\n")
        f.write(f"Sensitivity: {sensitivity:.4f}\n")
        f.write(f"Specificity: {specificity:.4f}\n")
        f.write(f"PPV (Precision): {ppv:.4f}\n")
        f.write(f"NPV: {npv:.4f}\n")
        f.write(f"F1-score: {f1:.4f}\n\n")
        f.write(f"Confusion Matrix:\n{conf_matrix}\n")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    classes = ['No Contraction', 'Contraction']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Add text labels
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, format(conf_matrix[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if conf_matrix[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    
    return {
        'accuracy': accuracy,
        'auc': roc_auc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv,
        'npv': npv,
        'f1': f1,
        'confusion_matrix': conf_matrix.tolist()
    }

def calculate_epoch_metrics(model, data_loader, device, mask_batch=None):
    """Calculate metrics on a dataset during training"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for X_batch, y_batch, mask_batch, len_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            mask_batch = mask_batch.to(device)
            
            # Get predictions
            y_pred = model(X_batch, len_batch)
            
            # Collect only valid predictions (non-padded)
            for i in range(len(X_batch)):
                valid_mask = mask_batch[i].bool()
                if valid_mask.sum() > 0:
                    all_labels.extend(y_batch[i][valid_mask].cpu().numpy())
                    all_probs.extend(y_pred[i][valid_mask].cpu().numpy())
                    all_preds.extend((y_pred[i][valid_mask] > 0.5).int().cpu().numpy())
    
    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    # Calculate basic metrics
    if len(np.unique(all_labels)) > 1:  # Check if we have both classes
        report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)
        conf_matrix = confusion_matrix(all_labels, all_preds)
        
        TN = conf_matrix[0, 0] if conf_matrix.shape == (2, 2) else 0
        FP = conf_matrix[0, 1] if conf_matrix.shape == (2, 2) else 0
        FN = conf_matrix[1, 0] if conf_matrix.shape == (2, 2) else 0
        TP = conf_matrix[1, 1] if conf_matrix.shape == (2, 2) else 0
        
        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0
        f1 = report['1']['f1-score'] if '1' in report else 0.0
    else:
        accuracy = 0.0
        roc_auc = 0.5
        f1 = 0.0
    
    return {
        'accuracy': accuracy,
        'auc': roc_auc,
        'f1': f1
    }

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train CNN-LSTM on sequential contraction data')
    parser.add_argument('--data_dir', type=str, default='data/mit-innaccel',
                       help='Directory containing data')
    parser.add_argument('--feature_set', type=str, default='top50',
                       choices=['top50', 'top600', 'full'],
                       help='Feature set to use')
    parser.add_argument('--output_dir', type=str, default='cnn_lstm_results',
                       help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--hidden_size', type=int, default=128,
                       help='LSTM hidden size')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate')
    parser.add_argument('--cnn_filters', type=str, default='64,128',
                       help='Number of filters for CNN layers (comma-separated)')
    parser.add_argument('--kernel_sizes', type=str, default='3,5',
                       help='Kernel sizes for CNN layers (comma-separated)')
    
    args = parser.parse_args()
    
    # Set up output directory
    if args.feature_set == 'top50':
        data_path = os.path.join(args.data_dir, 'sequential_top50_selected_features')
    elif args.feature_set == 'top600':
        data_path = os.path.join(args.data_dir, 'sequential_top600_selected_features')
    else:  # full
        data_path = os.path.join(args.data_dir, 'sequential_contraction_data')
    
    output_dir = os.path.join(args.output_dir, args.feature_set)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading data from {data_path}")
    X_train_raw = np.load(os.path.join(data_path, 'X_train.npy'))
    X_test_raw = np.load(os.path.join(data_path, 'X_test.npy'))
    y_train_raw = np.load(os.path.join(data_path, 'y_train.npy'))
    y_test_raw = np.load(os.path.join(data_path, 'y_test.npy'))
    
    print(f"Data shapes: X_train {X_train_raw.shape}, y_train {y_train_raw.shape}")
    print(f"Data shapes: X_test {X_test_raw.shape}, y_test {y_test_raw.shape}")
    
    # Handle NaN values immediately
    print("Preprocessing data to handle NaN values...")
    
    # Create masks for valid windows (not NaN in either X or y)
    train_x_valid = ~np.isnan(X_train_raw).any(axis=2)  # True where all features are valid
    train_y_valid = ~np.isnan(y_train_raw)  # True where y is valid
    train_valid = train_x_valid & train_y_valid  # Both X and y must be valid
    
    test_x_valid = ~np.isnan(X_test_raw).any(axis=2)
    test_y_valid = ~np.isnan(y_test_raw)
    test_valid = test_x_valid & test_y_valid
    
    print(f"Train valid windows: {np.sum(train_valid)}/{train_valid.size}")
    print(f"Test valid windows: {np.sum(test_valid)}/{test_valid.size}")
    
    # Replace remaining NaNs in X with zeros (shouldn't be any after masking, but just to be safe)
    X_train = np.copy(X_train_raw)
    X_train[np.isnan(X_train)] = 0
    
    X_test = np.copy(X_test_raw)
    X_test[np.isnan(X_test)] = 0
    
    # Replace NaNs in y with zeros (will be masked out anyway)
    y_train = np.copy(y_train_raw)
    y_train[np.isnan(y_train)] = 0
    
    y_test = np.copy(y_test_raw)
    y_test[np.isnan(y_test)] = 0
    
    print("Successfully replaced all NaN values with 0 and created valid masks")
    
    # Split train into train/val (80/20 of original train)
    num_train = int(0.8 * len(X_train))
    indices = np.random.permutation(len(X_train))
    train_idx, val_idx = indices[:num_train], indices[num_train:]
    
    X_val = X_train[val_idx].copy()
    y_val = y_train[val_idx].copy()
    val_masks = train_valid[val_idx].copy()
    
    X_train = X_train[train_idx].copy()
    y_train = y_train[train_idx].copy()
    train_masks = train_valid[train_idx].copy()
    
    print(f"Split train/val: X_train {X_train.shape}, X_val {X_val.shape}")
    
    # Create datasets with explicit array copying to avoid stride issues
    print("Creating datasets with explicit array copying...")
    
    # First convert everything to numpy arrays with the right memory layout
    X_train_fixed = np.array(X_train, copy=True, order='C')
    y_train_fixed = np.array(y_train, copy=True, order='C')
    train_masks_fixed = np.array(train_masks, copy=True, order='C')
    
    X_val_fixed = np.array(X_val, copy=True, order='C')
    y_val_fixed = np.array(y_val, copy=True, order='C')
    val_masks_fixed = np.array(val_masks, copy=True, order='C')
    
    X_test_fixed = np.array(X_test, copy=True, order='C')
    y_test_fixed = np.array(y_test, copy=True, order='C')
    test_valid_fixed = np.array(test_valid, copy=True, order='C')
    
    # Create datasets
    train_dataset = SequenceDataset(X_train_fixed, y_train_fixed, train_masks_fixed)
    val_dataset = SequenceDataset(X_val_fixed, y_val_fixed, val_masks_fixed)
    test_dataset = SequenceDataset(X_test_fixed, y_test_fixed, test_valid_fixed)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Parse CNN parameters
    cnn_filters = [int(x) for x in args.cnn_filters.split(',')]
    kernel_sizes = [int(x) for x in args.kernel_sizes.split(',')]
    
    # Create model
    input_size = X_train.shape[2]  # Number of features
    model = CNN_LSTM(
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        cnn_filters=cnn_filters,
        kernel_sizes=kernel_sizes,
        dropout=args.dropout
    )
    model = model.to(device)
    
    # Print model summary
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {num_params} parameters")
    print(f"CNN filters: {cnn_filters}, Kernel sizes: {kernel_sizes}")
    
    # Train model
    print("Starting training...")
    start_time = time.time()
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        output_dir=output_dir
    )
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluate model
    print("Evaluating model...")
    eval_dir = os.path.join(output_dir, 'evaluation')
    metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        output_dir=eval_dir
    )
    
    print("Results:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"Results saved to {eval_dir}")

if __name__ == "__main__":
    main()
