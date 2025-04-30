#!/usr/bin/env python3
import os
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split

def load_selected_features():
    """Load the selected features from the JSON file"""
    with open('analysis_results/selected_features.json', 'r') as f:
        selected_features = json.load(f)
    return selected_features

def load_original_data():
    """Load the original feature matrix and names"""
    X = np.load('data/final_contraction_data/X.npy')
    y = np.load('data/final_contraction_data/y.npy')
    feature_names = np.load('data/final_contraction_data/feature_names.npy', allow_pickle=True)
    return X, y, feature_names

def create_selected_features_data():
    """Create a new dataset with only the selected features and perform train-test split"""
    # Load the selected features
    selected_features = load_selected_features()
    
    # Load the original data
    X, y, feature_names = load_original_data()
    
    # Get indices of selected features
    selected_indices = [np.where(feature_names == feature)[0][0] for feature in selected_features]
    
    # Extract selected features
    X_selected = X[:, selected_indices]
    feature_names_selected = feature_names[selected_indices]
    
    # Perform train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create output directory if it doesn't exist
    output_dir = 'data/selected_features_data'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the split data
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
    np.save(os.path.join(output_dir, 'feature_names.npy'), feature_names_selected)
    
    # Copy other necessary files
    for file in ['dataset_info.npy']:
        if os.path.exists(os.path.join('data/contraction_data', file)):
            os.system(f'cp {os.path.join("data/contraction_data", file)} {os.path.join(output_dir, file)}')
    
    print(f"Created new dataset with {len(selected_features)} features")
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Saved to {output_dir}")

if __name__ == "__main__":
    create_selected_features_data() 