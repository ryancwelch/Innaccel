#!/usr/bin/env python
"""
Train a Transformer model on sequential contraction data.
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
from models.Transformer import TransformerModel
import math

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

def train_model(model, train_loader, val_loader, device, epochs=30, 
                learning_rate=0.001, weight_decay=1e-5, output_dir='model_output',
                gradient_accumulation_steps=1):
    """Train the model and save results"""
    os.makedirs(output_dir, exist_ok=True)
    model_dir = os.path.join(output_dir, 'model')
    os.makedirs(model_dir, exist_ok=True)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Track metrics
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    no_improve_count = 0
    best_model_path = os.path.join(model_dir, 'best_model.pt')
    
    print(f"Using gradient accumulation with {gradient_accumulation_steps} steps")
    
    # Training loop
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_samples = 0
        
        # Reset gradients at the beginning of each epoch
        optimizer.zero_grad()
        
        # Batch counter for gradient accumulation
        batch_count = 0
        accumulated_loss = 0
        
        for X_batch, y_batch, mask_batch, len_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            mask_batch = mask_batch.to(device)
            
            # Create key padding mask for transformer (True where padding)
            key_padding_mask = ~mask_batch
            
            # Forward pass
            try:
                y_pred = model(X_batch, src_key_padding_mask=key_padding_mask)
                
                # Ensure output has same shape as target
                if y_pred.size() != y_batch.size():
                    print(f"Warning: Model output shape {y_pred.size()} doesn't match target shape {y_batch.size()}")
                    # If output is a single value per sequence, repeat to match sequence length
                    if len(y_pred.size()) == 1:
                        y_pred = y_pred.unsqueeze(1).repeat(1, y_batch.size(1))
                    # If output has an extra dimension, squeeze it
                    elif len(y_pred.size()) > len(y_batch.size()):
                        y_pred = y_pred.squeeze(-1)
                
                # Check for NaN in predictions and replace with 0.5
                if torch.isnan(y_pred).any():
                    print("Warning: NaN detected in predictions, replacing with 0.5")
                    y_pred = torch.where(torch.isnan(y_pred), torch.tensor(0.5, device=device), y_pred)
                
                # Calculate weighted binary cross entropy manually
                # Only consider non-padded elements
                y_ones = y_batch == 1
                y_zeros = (y_batch == 0) & mask_batch  # Only masked zeros
                
                # Count positives and negatives to compute class weights
                num_pos = torch.sum(y_ones).item()
                num_neg = torch.sum(y_zeros).item()
                total = num_pos + num_neg
                
                if total > 0 and num_pos > 0 and num_neg > 0:
                    pos_weight = total / (2 * num_pos)
                    neg_weight = total / (2 * num_neg)
                else:
                    pos_weight = 1.0
                    neg_weight = 1.0
                
                # Focal loss parameters
                gamma = 2.0  # Focusing parameter (2.0 is common)
                
                # Calculate focal loss with numerical stability
                # Add epsilon for stability
                epsilon = 1e-7
                y_pred = torch.clamp(y_pred, epsilon, 1.0 - epsilon)
                
                pt_pos = y_pred
                pt_neg = 1 - y_pred
                
                # (1-pt)^gamma * log(pt)
                focal_pos = torch.pow(1 - pt_pos, gamma) * torch.log(pt_pos)
                focal_neg = torch.pow(1 - pt_neg, gamma) * torch.log(pt_neg)
                
                # Combine with class weights
                bce_pos = -focal_pos * y_batch
                bce_neg = -focal_neg * (1 - y_batch)
                
                # Apply weights and masking
                weighted_bce = pos_weight * bce_pos + neg_weight * bce_neg
                masked_bce = weighted_bce * mask_batch
                
                # Average loss
                loss = torch.sum(masked_bce) / torch.sum(mask_batch)
                
                # Check for NaN in loss
                if torch.isnan(loss).any():
                    print("Warning: NaN detected in loss, skipping batch")
                    continue
                
                # Normalize loss for gradient accumulation
                loss = loss / gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
                
                # Track metrics (use the un-normalized loss)
                batch_loss = loss.item() * gradient_accumulation_steps
                accumulated_loss += batch_loss
                train_loss += batch_loss * torch.sum(mask_batch).item()
                train_samples += torch.sum(mask_batch).item()
                
                # Increment batch counter
                batch_count += 1
                
                # Update weights only after accumulating gradients
                if batch_count % gradient_accumulation_steps == 0:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                    
                    # Update weights
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    # Print progress
                    print(f"Epoch {epoch+1}, Batch {batch_count//gradient_accumulation_steps}: Accumulated Loss = {accumulated_loss:.4f}")
                    accumulated_loss = 0
            
            except RuntimeError as e:
                print(f"Error during training: {e}")
                print("Skipping this batch and continuing...")
                continue
        
        # Update weights for any remaining gradients
        if batch_count % gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            optimizer.zero_grad()
        
        # Calculate epoch loss
        epoch_train_loss = train_loss / train_samples if train_samples > 0 else 0
        train_losses.append(epoch_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        val_samples = 0
        valid_val_batches = 0
        
        with torch.no_grad():
            for X_batch, y_batch, mask_batch, len_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                mask_batch = mask_batch.to(device)
                
                # Create key padding mask for transformer (True where padding)
                key_padding_mask = ~mask_batch
                
                try:
                    # Forward pass
                    y_pred = model(X_batch, src_key_padding_mask=key_padding_mask)
                    
                    # Ensure output has same shape as target
                    if y_pred.size() != y_batch.size():
                        # If output is a single value per sequence, repeat to match sequence length
                        if len(y_pred.size()) == 1:
                            y_pred = y_pred.unsqueeze(1).repeat(1, y_batch.size(1))
                        # If output has an extra dimension, squeeze it
                        elif len(y_pred.size()) > len(y_batch.size()):
                            y_pred = y_pred.squeeze(-1)
                    
                    # Check for NaN in predictions and replace with 0.5
                    if torch.isnan(y_pred).any():
                        print("Warning: NaN detected in predictions during validation, replacing with 0.5")
                        y_pred = torch.where(torch.isnan(y_pred), torch.tensor(0.5, device=device), y_pred)
                    
                    # Calculate loss (same as training)
                    y_ones = y_batch == 1
                    y_zeros = (y_batch == 0) & mask_batch
                    
                    num_pos = torch.sum(y_ones).item()
                    num_neg = torch.sum(y_zeros).item()
                    total = num_pos + num_neg
                    
                    if total > 0 and num_pos > 0 and num_neg > 0:
                        pos_weight = total / (2 * num_pos)
                        neg_weight = total / (2 * num_neg)
                    else:
                        pos_weight = 1.0
                        neg_weight = 1.0
                    
                    # Focal loss parameters
                    gamma = 2.0  # Focusing parameter (2.0 is common)
                    
                    # Calculate focal loss with numerical stability
                    # Add epsilon for stability
                    epsilon = 1e-7
                    y_pred = torch.clamp(y_pred, epsilon, 1.0 - epsilon)
                    
                    pt_pos = y_pred
                    pt_neg = 1 - y_pred
                    
                    # (1-pt)^gamma * log(pt)
                    focal_pos = torch.pow(1 - pt_pos, gamma) * torch.log(pt_pos)
                    focal_neg = torch.pow(1 - pt_neg, gamma) * torch.log(pt_neg)
                    
                    # Combine with class weights
                    bce_pos = -focal_pos * y_batch
                    bce_neg = -focal_neg * (1 - y_batch)
                    
                    weighted_bce = pos_weight * bce_pos + neg_weight * bce_neg
                    masked_bce = weighted_bce * mask_batch
                    
                    batch_loss = torch.sum(masked_bce) / torch.sum(mask_batch) if torch.sum(mask_batch) > 0 else 0
                    
                    # Check for NaN in loss
                    if not torch.isnan(batch_loss).any():
                        val_loss += batch_loss.item() * torch.sum(mask_batch).item()
                        val_samples += torch.sum(mask_batch).item()
                        valid_val_batches += 1
                    else:
                        print("Warning: NaN detected in validation loss, skipping batch")
                
                except RuntimeError as e:
                    print(f"Error during validation: {e}")
                    print("Skipping this batch and continuing...")
                    continue
        
        # Calculate epoch validation loss
        epoch_val_loss = val_loss / val_samples if val_samples > 0 else float('inf')
        val_losses.append(epoch_val_loss)
        
        # Print progress
        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}')
        
        # Save history at each epoch for real-time monitoring
        with open(os.path.join(output_dir, 'training_progress.txt'), 'a') as f:
            f.write(f'Epoch {epoch+1}: Train Loss = {epoch_train_loss:.6f}, Val Loss = {epoch_val_loss:.6f}\n')
        
        # Calculate and report metrics on validation set for monitoring
        if valid_val_batches > 0 and ((epoch + 1) % 5 == 0 or epoch == 0):  # Every 5 epochs and at the start
            # Get validation metrics
            try:
                val_metrics = calculate_epoch_metrics(model, val_loader, device)
                print(f'  Val Metrics - Acc: {val_metrics["accuracy"]:.4f}, AUC: {val_metrics["auc"]:.4f}, F1: {val_metrics["f1"]:.4f}')
                
                with open(os.path.join(output_dir, 'training_progress.txt'), 'a') as f:
                    f.write(f'  Val Metrics - Acc: {val_metrics["accuracy"]:.4f}, AUC: {val_metrics["auc"]:.4f}, F1: {val_metrics["f1"]:.4f}\n')
            except Exception as e:
                print(f"Error calculating validation metrics: {e}")
        
        # Update learning rate
        if not math.isnan(epoch_val_loss) and not math.isinf(epoch_val_loss):
            scheduler.step(epoch_val_loss)
        
        # Save best model
        if epoch_val_loss < best_val_loss and not math.isnan(epoch_val_loss) and not math.isinf(epoch_val_loss):
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f'Saved new best model with validation loss: {best_val_loss:.4f}')
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= 5:  # Early stopping
                print('Early stopping triggered')
                break
    
    # Save final model
    final_model_path = os.path.join(model_dir, 'final_model.pt')
    torch.save(model.state_dict(), final_model_path)
    
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
    
    # Load best model for evaluation if it exists
    try:
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path))
            print(f"Loaded best model from {best_model_path}")
        else:
            print(f"Best model not found at {best_model_path}, using final model")
    except Exception as e:
        print(f"Error loading best model: {e}")
    
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
            
            # Create key padding mask for transformer (True where padding)
            key_padding_mask = ~mask_batch
            
            try:
                # Get predictions
                y_pred = model(X_batch, src_key_padding_mask=key_padding_mask)
                
                # Ensure output has same shape as target
                if y_pred.size() != y_batch.size():
                    # If output is a single value per sequence, repeat to match sequence length
                    if len(y_pred.size()) == 1:
                        y_pred = y_pred.unsqueeze(1).repeat(1, y_batch.size(1))
                    # If output has an extra dimension, squeeze it
                    elif len(y_pred.size()) > len(y_batch.size()):
                        y_pred = y_pred.squeeze(-1)
                
                # Check for NaN in predictions and replace with 0.5
                if torch.isnan(y_pred).any():
                    print("Warning: NaN detected in test predictions, replacing with 0.5")
                    y_pred = torch.where(torch.isnan(y_pred), torch.tensor(0.5, device=device), y_pred)
                
                # Collect only valid predictions (non-padded)
                for i in range(len(X_batch)):
                    valid_mask = mask_batch[i].bool()
                    if valid_mask.sum() > 0:
                        batch_labels = y_batch[i][valid_mask].cpu().numpy()
                        batch_probs = y_pred[i][valid_mask].cpu().numpy()
                        batch_preds = (y_pred[i][valid_mask] > 0.5).int().cpu().numpy()
                        
                        # Filter out any NaN values that might still exist
                        valid_idx = np.where(~np.isnan(batch_probs) & ~np.isinf(batch_probs))[0]
                        if len(valid_idx) > 0:
                            all_labels.extend(batch_labels[valid_idx])
                            all_probs.extend(batch_probs[valid_idx])
                            all_preds.extend(batch_preds[valid_idx])
            except RuntimeError as e:
                print(f"Error during evaluation: {e}")
                continue
    
    # Convert to numpy arrays
    if not all_labels:
        print("Warning: No valid predictions for evaluation")
        return {
            'accuracy': 0.0,
            'auc': 0.5,
            'sensitivity': 0.0,
            'specificity': 0.0,
            'ppv': 0.0,
            'npv': 0.0,
            'f1': 0.0,
            'confusion_matrix': [[0, 0], [0, 0]]
        }
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    try:
        report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
        
        # Calculate confusion matrix
        conf_matrix = confusion_matrix(all_labels, all_preds)
        
        # Handle case where there's only one class in predictions
        if conf_matrix.shape == (1, 1):
            # Expand to 2x2 with zeros
            if all_preds[0] == 0:  # Only negative predictions
                conf_matrix = np.array([[conf_matrix[0, 0], 0], [0, 0]])
            else:  # Only positive predictions
                conf_matrix = np.array([[0, 0], [0, conf_matrix[0, 0]]])
        
        # Calculate AUC
        try:
            fpr, tpr, _ = roc_curve(all_labels, all_probs)
            roc_auc = auc(fpr, tpr)
        except:
            print("Warning: Could not calculate ROC AUC. Using default value of 0.5")
            roc_auc = 0.5
            fpr = np.array([0, 1])
            tpr = np.array([0, 1])
        
        # Calculate additional metrics
        TN = conf_matrix[0, 0]
        FP = conf_matrix[0, 1]
        FN = conf_matrix[1, 0] if conf_matrix.shape == (2, 2) else 0
        TP = conf_matrix[1, 1] if conf_matrix.shape == (2, 2) else 0
        
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        ppv = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        npv = TN / (TN + FN) if (TN + FN) > 0 else 0.0
        f1 = report['1']['f1-score'] if '1' in report else 0.0
        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0
    
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        # Set default values
        accuracy = 0.0
        roc_auc = 0.5
        sensitivity = 0.0
        specificity = 0.0
        ppv = 0.0
        npv = 0.0
        f1 = 0.0
        conf_matrix = np.array([[0, 0], [0, 0]])
        fpr = np.array([0, 1])
        tpr = np.array([0, 1])
    
    # Save results
    with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
        f.write("Transformer Model Evaluation Results\n")
        f.write("===================================\n\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"ROC AUC: {roc_auc:.4f}\n")
        f.write(f"Sensitivity: {sensitivity:.4f}\n")
        f.write(f"Specificity: {specificity:.4f}\n")
        f.write(f"PPV (Precision): {ppv:.4f}\n")
        f.write(f"NPV: {npv:.4f}\n")
        f.write(f"F1-score: {f1:.4f}\n\n")
        f.write(f"Confusion Matrix:\n{conf_matrix}\n")
    
    # Plot confusion matrix
    try:
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
    except Exception as e:
        print(f"Error plotting confusion matrix: {e}")
    
    # Plot ROC curve
    try:
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
    except Exception as e:
        print(f"Error plotting ROC curve: {e}")
    
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

def calculate_epoch_metrics(model, data_loader, device):
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
            
            # Create key padding mask for transformer (True where padding)
            key_padding_mask = ~mask_batch
            
            # Get predictions
            try:
                y_pred = model(X_batch, src_key_padding_mask=key_padding_mask)
                
                # Ensure output has same shape as target
                if y_pred.size() != y_batch.size():
                    # If output is a single value per sequence, repeat to match sequence length
                    if len(y_pred.size()) == 1:
                        y_pred = y_pred.unsqueeze(1).repeat(1, y_batch.size(1))
                    # If output has an extra dimension, squeeze it
                    elif len(y_pred.size()) > len(y_batch.size()):
                        y_pred = y_pred.squeeze(-1)
                
                # Check for NaN in predictions and replace with 0.5
                if torch.isnan(y_pred).any():
                    y_pred = torch.where(torch.isnan(y_pred), torch.tensor(0.5, device=device), y_pred)
                
                # Collect only valid predictions (non-padded)
                for i in range(len(X_batch)):
                    valid_mask = mask_batch[i].bool()
                    if valid_mask.sum() > 0:
                        batch_labels = y_batch[i][valid_mask].cpu().numpy()
                        batch_probs = y_pred[i][valid_mask].cpu().numpy()
                        batch_preds = (y_pred[i][valid_mask] > 0.5).int().cpu().numpy()
                        
                        # Check for NaN or Inf values
                        valid_idx = np.where(~np.isnan(batch_probs) & ~np.isinf(batch_probs))[0]
                        if len(valid_idx) > 0:
                            all_labels.extend(batch_labels[valid_idx])
                            all_probs.extend(batch_probs[valid_idx])
                            all_preds.extend(batch_preds[valid_idx])
            
            except RuntimeError as e:
                print(f"Error during metrics calculation: {e}")
                continue
    
    # Convert to numpy arrays
    if not all_labels:
        print("Warning: No valid predictions for metrics calculation")
        return {"accuracy": 0.0, "auc": 0.5, "f1": 0.0}
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    # Calculate basic metrics
    if len(np.unique(all_labels)) > 1 and len(all_labels) > 1:  # Check if we have both classes
        try:
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
        except Exception as e:
            print(f"Error in metrics calculation: {e}")
            accuracy = 0.0
            roc_auc = 0.5
            f1 = 0.0
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
    parser = argparse.ArgumentParser(description='Train a Transformer model on sequential data.')
    parser.add_argument('--data_dir', type=str, default='data/mit-innaccel', help='Directory containing data files')
    parser.add_argument('--feature_set', type=str, choices=['top50', 'top600', 'full'], default='top50', help='Feature set to use')
    parser.add_argument('--output_dir', type=str, default='transformer_results', help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--d_model', type=int, default=64, help='Dimension of model')
    parser.add_argument('--nhead', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of transformer layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--max_seq_length', type=int, default=1500, help='Maximum sequence length for positional encoding')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Number of gradient accumulation steps')
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
    
    # Get embedding dimension
    d_model = args.d_model
    
    # Create model
    model = TransformerModel(
        input_size=X_train.shape[2],  # Number of features
        d_model=d_model,
        num_layers=args.num_layers,
        dropout=args.dropout,
        max_seq_length=args.max_seq_length
    )
    model = model.to(device)
    
    # Print model summary
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {num_params} parameters")
    
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
        weight_decay=args.weight_decay,
        output_dir=output_dir,
        gradient_accumulation_steps=args.gradient_accumulation_steps
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
