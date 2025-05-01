#!/usr/bin/env python3
import os
import numpy as np
import joblib
import json
import argparse
from sklearn.model_selection import train_test_split

def load_selected_features(selected_features_path):
    """Load the selected features from the JSON file"""
    with open(selected_features_path, 'r') as f:
        selected_features = json.load(f)
    return selected_features

def load_original_data():
    """Load the original feature matrix and names"""
    X = np.load('data/sequential_contraction_data/X.npy')
    y = np.load('data/sequential_contraction_data/y.npy')
    feature_names = np.load('data/final_contraction_data/feature_names.npy', allow_pickle=True)
    return X, y, feature_names

def create_selected_features_data(output_dir, selected_features_path):
    """Create a new dataset with only the selected features and perform train-test split"""
    # Load the selected features
    selected_features = load_selected_features(selected_features_path)
    
    # Load the original data
    X, y, feature_names = load_original_data()
    
    # Get indices of selected features
    selected_indices = [np.where(feature_names == feature)[0][0] for feature in selected_features]
    
    # Extract selected features
    X_selected = X[:, :, selected_indices]
    feature_names_selected = feature_names[selected_indices]
    
    # Perform train-test split along records
    n_records = X_selected.shape[0]
    record_indices = np.arange(n_records)
    
    train_indices, test_indices = train_test_split(
        record_indices, test_size=0.2, random_state=42
    )
    
    # Split the data
    X_train = X_selected[train_indices]
    X_test = X_selected[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the split data
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
    np.save(os.path.join(output_dir, 'feature_names.npy'), feature_names_selected)
    
    # # Copy other necessary files
    # for file in ['dataset_info.npy']:
    #     if os.path.exists(os.path.join('data/contraction_data', file)):
    #         os.system(f'cp {os.path.join("data/contraction_data", file)} {os.path.join(output_dir, file)}')
    
    print(f"Created new dataset with {len(selected_features)} features")
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract selected features and create train-test split')
    parser.add_argument('--output-dir', type=str, default='data/sequential_top600_selected_features',
                        help='Directory to save the output files (default: data/sequential_top600_selected_features)')
    parser.add_argument('--selected-features', type=str, default='analysis_results/selected_features.json',
                        help='Path to the JSON file containing selected features (default: analysis_results/selected_features.json)')
    
    args = parser.parse_args()
    create_selected_features_data(args.output_dir, args.selected_features) 