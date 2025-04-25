#!/usr/bin/env python
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from tqdm import tqdm
import argparse
import pickle
import joblib

def load_dataset(data_dir="data/contraction_data"):
    """
    Load the prepared contraction detection dataset.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing the dataset
        
    Returns:
    --------
    X_train : np.ndarray
        Training features
    X_test : np.ndarray
        Testing features
    y_train : np.ndarray
        Training labels
    y_test : np.ndarray
        Testing labels
    feature_names : list
        List of feature names
    dataset_info : dict
        Information about the dataset
    """
    try:
        # Check if train/test splits exist
        if os.path.exists(os.path.join(data_dir, 'X_train.npy')):
            X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
            X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
            y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
            y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
            feature_names = np.load(os.path.join(data_dir, 'feature_names.npy'), allow_pickle=True)
            dataset_info = np.load(os.path.join(data_dir, 'dataset_info.npy'), allow_pickle=True).item()
            
            print(f"Loaded train/test splits from {data_dir}")
            print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
            
            return X_train, X_test, y_train, y_test, feature_names, dataset_info
        
        # Otherwise load full dataset
        X = np.load(os.path.join(data_dir, 'X.npy'))
        y = np.load(os.path.join(data_dir, 'y.npy'))
        feature_names = np.load(os.path.join(data_dir, 'feature_names.npy'), allow_pickle=True)
        dataset_info = np.load(os.path.join(data_dir, 'dataset_info.npy'), allow_pickle=True).item()
        
        print(f"Loaded full dataset from {data_dir}")
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        
        # Create a simple train/test split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Created train/test split: {X_train.shape}, {X_test.shape}")
        
        return X_train, X_test, y_train, y_test, feature_names, dataset_info
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None, None, None, None, None

def preprocess_data(X_train, X_test, normalize=True):
    """
    Preprocess the data, normalizing if requested.
    
    Parameters:
    -----------
    X_train : np.ndarray
        Training features
    X_test : np.ndarray
        Testing features
    normalize : bool
        Whether to normalize the data
        
    Returns:
    --------
    X_train_proc : np.ndarray
        Processed training features
    X_test_proc : np.ndarray
        Processed testing features
    preprocessing_info : dict
        Information about preprocessing
    """
    preprocessing_info = {}
    
    # Handle missing values
    X_train_proc = np.nan_to_num(X_train)
    X_test_proc = np.nan_to_num(X_test)
    
    # Normalize if requested
    if normalize:
        # Calculate mean and std on training data
        mean = np.mean(X_train_proc, axis=0)
        std = np.std(X_train_proc, axis=0)
        
        # Avoid division by zero
        std[std == 0] = 1.0
        
        # Normalize
        X_train_proc = (X_train_proc - mean) / std
        X_test_proc = (X_test_proc - mean) / std
        
        preprocessing_info['mean'] = mean
        preprocessing_info['std'] = std
    
    preprocessing_info['normalize'] = normalize
    
    return X_train_proc, X_test_proc, preprocessing_info

def train_random_forest(X_train, y_train, hyperparams=None):
    """
    Train a Random Forest model on the data.
    
    Parameters:
    -----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training labels
    hyperparams : dict
        Hyperparameters for the model
        
    Returns:
    --------
    model : RandomForestClassifier
        Trained model
    """
    if hyperparams is None:
        hyperparams = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42,
            'class_weight': 'balanced'
        }
    
    print(f"Training Random Forest with parameters: {hyperparams}")
    model = RandomForestClassifier(**hyperparams)
    model.fit(X_train, y_train)
    
    return model

def train_xgboost(X_train, y_train, hyperparams=None):
    """
    Train an XGBoost model on the data.
    
    Parameters:
    -----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training labels
    hyperparams : dict
        Hyperparameters for the model
        
    Returns:
    --------
    model : xgb.XGBClassifier
        Trained model
    """
    if hyperparams is None:
        hyperparams = {
            'n_estimators': 100,
            'max_depth': 3,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'scale_pos_weight': np.sum(y_train == 0) / np.sum(y_train == 1)
        }
    
    print(f"Training XGBoost with parameters: {hyperparams}")
    model = xgb.XGBClassifier(**hyperparams)
    model.fit(X_train, y_train)
    
    return model

def train_with_hyperparameter_tuning(X_train, y_train, model_type='random_forest'):
    """
    Train a model with hyperparameter tuning.
    
    Parameters:
    -----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training labels
    model_type : str
        Type of model to train ('random_forest' or 'xgboost')
        
    Returns:
    --------
    best_model : sklearn model
        Trained model with best parameters
    """
    from sklearn.model_selection import RandomizedSearchCV
    
    if model_type == 'random_forest':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': ['balanced', 'balanced_subsample', None]
        }
        model = RandomForestClassifier(random_state=42)
        
    elif model_type == 'xgboost':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'scale_pos_weight': [1, 2, 5, np.sum(y_train == 0) / max(1, np.sum(y_train == 1))]
        }
        model = xgb.XGBClassifier(random_state=42)
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    print(f"Starting randomized search for {model_type}...")
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=20,
        cv=3,
        scoring='f1',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    random_search.fit(X_train, y_train)
    
    print(f"Best parameters: {random_search.best_params_}")
    print(f"Best score: {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_

def evaluate_model(model, X_test, y_test, feature_names=None, output_dir=None):
    """
    Evaluate a trained model on test data.
    
    Parameters:
    -----------
    model : sklearn model
        Trained model
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test labels
    feature_names : list
        List of feature names
    output_dir : str
        Directory to save evaluation results
        
    Returns:
    --------
    metrics : dict
        Evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    metrics = {}
    
    # Classification report
    class_report = classification_report(y_test, y_pred, output_dict=True)
    metrics['classification_report'] = class_report
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    metrics['confusion_matrix'] = conf_matrix
    
    # Print metrics
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(conf_matrix)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    classes = ['No Contraction', 'Contraction']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Add labels
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, format(conf_matrix[i, j], 'd'),
                   horizontalalignment="center",
                   color="white" if conf_matrix[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    # ROC curve and AUC
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        metrics['roc'] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    # Feature importance
    if hasattr(model, 'feature_importances_') and feature_names is not None:
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        top_n = min(20, len(feature_names))
        
        metrics['feature_importance'] = {
            'feature_names': [feature_names[i] for i in indices[:top_n]],
            'importance': importances[indices[:top_n]]
        }
        
        plt.figure(figsize=(12, 8))
        plt.title("Feature Importances (Top 20)")
        plt.bar(range(top_n), importances[indices[:top_n]], align="center")
        plt.xticks(range(top_n), [feature_names[i] for i in indices[:top_n]], rotation=90)
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    return metrics

def save_model(model, preprocessing_info, output_dir, model_name="contraction_model"):
    """
    Save a trained model and its preprocessing information.
    
    Parameters:
    -----------
    model : sklearn model
        Trained model
    preprocessing_info : dict
        Information about preprocessing
    output_dir : str
        Directory to save the model
    model_name : str
        Name of the model
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, f"{model_name}.joblib")
    joblib.dump(model, model_path)
    
    # Save preprocessing info
    preproc_path = os.path.join(output_dir, f"{model_name}_preprocessing.joblib")
    joblib.dump(preprocessing_info, preproc_path)
    
    print(f"Model saved to {model_path}")
    print(f"Preprocessing info saved to {preproc_path}")

def main():
    parser = argparse.ArgumentParser(
        description='Train a machine learning model for contraction detection',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--data_dir', type=str, default='data/contraction_data',
                      help='Directory containing the prepared dataset')
    parser.add_argument('--output_dir', type=str, default='models',
                      help='Directory to save trained models and results')
    parser.add_argument('--model_type', type=str, default='random_forest', 
                      choices=['random_forest', 'xgboost'],
                      help='Type of model to train')
    parser.add_argument('--tune_hyperparams', action='store_true',
                      help='Perform hyperparameter tuning')
    parser.add_argument('--normalize', action='store_true',
                      help='Normalize features')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    X_train, X_test, y_train, y_test, feature_names, dataset_info = load_dataset(args.data_dir)
    
    if X_train is None:
        print("Failed to load dataset. Exiting.")
        sys.exit(1)
    
    # Preprocess data
    X_train_proc, X_test_proc, preprocessing_info = preprocess_data(
        X_train, X_test, normalize=args.normalize
    )
    
    # Train model
    if args.tune_hyperparams:
        model = train_with_hyperparameter_tuning(X_train_proc, y_train, model_type=args.model_type)
    else:
        if args.model_type == 'random_forest':
            model = train_random_forest(X_train_proc, y_train)
        elif args.model_type == 'xgboost':
            model = train_xgboost(X_train_proc, y_train)
        else:
            print(f"Unknown model type: {args.model_type}")
            sys.exit(1)
    
    # Evaluate model
    metrics = evaluate_model(model, X_test_proc, y_test, feature_names, args.output_dir)
    
    # Save model and results
    save_model(model, preprocessing_info, args.output_dir, 
              model_name=f"contraction_{args.model_type}")
    
    # Save metrics
    metrics_path = os.path.join(args.output_dir, f"contraction_{args.model_type}_metrics.json")
    import json
    with open(metrics_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        metrics_json = {}
        for key, value in metrics.items():
            if isinstance(value, dict):
                metrics_json[key] = {}
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        metrics_json[key][k] = v.tolist()
                    else:
                        metrics_json[key][k] = v
            else:
                if isinstance(value, np.ndarray):
                    metrics_json[key] = value.tolist()
                else:
                    metrics_json[key] = value
        
        json.dump(metrics_json, f, indent=2)
    
    print(f"Metrics saved to {metrics_path}")
    print(f"\nTraining and evaluation complete!")

if __name__ == "__main__":
    main() 