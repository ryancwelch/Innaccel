#!/usr/bin/env python
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import joblib
import argparse
from time import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from tqdm import tqdm
import warnings
from sklearn.base import clone
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek
from collections import Counter

# Inherit everything from train_evaluate_models.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_evaluate_models import *

# Override the necessary functions to handle class imbalance
def balance_dataset(X_train, y_train, method='smote', random_state=42):
    """
    Balance the dataset using various sampling techniques.
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Training feature data
    y_train : numpy.ndarray
        Training labels
    method : str
        Balancing method: 'smote', 'random_over', 'random_under', 'smote_enn', 'smote_tomek'
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    tuple
        (X_train_balanced, y_train_balanced)
    """
    print(f"Original class distribution: {Counter(y_train)}")
    
    if method == 'smote':
        sampler = SMOTE(random_state=random_state)
    elif method == 'random_over':
        sampler = RandomOverSampler(random_state=random_state)
    elif method == 'random_under':
        sampler = RandomUnderSampler(random_state=random_state)
    elif method == 'smote_enn':
        sampler = SMOTEENN(random_state=random_state)
    elif method == 'smote_tomek':
        sampler = SMOTETomek(random_state=random_state)
    else:
        raise ValueError(f"Unknown balancing method: {method}")
    
    X_train_balanced, y_train_balanced = sampler.fit_resample(X_train, y_train)
    
    print(f"Balanced class distribution: {Counter(y_train_balanced)}")
    return X_train_balanced, y_train_balanced

def train_model_balanced(model, dataset, tune_hyperparams=False, balance_method='smote'):
    """
    Train a model on a balanced version of the dataset.
    
    Parameters:
    -----------
    model : sklearn model
        Model to train
    dataset : dict
        Processed dataset dictionary
    tune_hyperparams : bool
        Whether to perform hyperparameter tuning
    balance_method : str
        Method to use for balancing the dataset
        
    Returns:
    --------
    model
        Trained model
    dict
        Training information
    """
    X_train = dataset['X_train']
    y_train = dataset['y_train']
    
    # Balance the training data
    X_train_balanced, y_train_balanced = balance_dataset(X_train, y_train, method=balance_method)
    
    # Create a validation set from the balanced data
    X_train_main, X_val, y_train_main, y_val = train_test_split(
        X_train_balanced, y_train_balanced, test_size=0.2, random_state=42, stratify=y_train_balanced
    )
    
    training_info = {
        'training_time': None,
        'best_params': None,
        'learning_curves': {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': []
        },
        'balance_method': balance_method,
        'original_class_distribution': dict(Counter(y_train)),
        'balanced_class_distribution': dict(Counter(y_train_balanced))
    }
    
    if tune_hyperparams:
        # Define parameter grids for different model types
        param_grids = {
            'rf': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'xgb': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0]
            },
            'lr': {
                'C': [0.01, 0.1, 1.0, 10.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'svm': {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto', 0.1, 0.01]
            },
            'mlp': {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive']
            }
        }
        
        # Get model type from model
        model_type = None
        if isinstance(model, RandomForestClassifier):
            model_type = 'rf'
        elif isinstance(model, xgb.XGBClassifier):
            model_type = 'xgb'
        elif isinstance(model, LogisticRegression):
            model_type = 'lr'
        elif isinstance(model, SVC):
            model_type = 'svm'
        elif isinstance(model, MLPClassifier):
            model_type = 'mlp'
        
        if model_type and model_type in param_grids:
            print(f"Performing hyperparameter tuning for {MODEL_TYPES.get(model_type, model_type)}")
            
            # Use RandomizedSearchCV for efficiency
            grid_search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_grids[model_type],
                n_iter=10,
                cv=3,
                scoring='f1',
                random_state=42,
                n_jobs=-1,
                verbose=1
            )
            
            start_time = time()
            grid_search.fit(X_train_balanced, y_train_balanced)
            training_time = time() - start_time
            
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best score: {grid_search.best_score_:.4f}")
            
            model = grid_search.best_estimator_
            training_info['best_params'] = grid_search.best_params_
            training_info['training_time'] = training_time
            
            return model, training_info
    
    # Regular training without hyperparameter tuning
    print(f"Training model on balanced data...")
    start_time = time()
    
    # Track learning curves based on model type
    if isinstance(model, xgb.XGBClassifier):
        # XGBoost has built-in eval_set functionality
        eval_set = [(X_train_main, y_train_main), (X_val, y_val)]
        model.fit(
            X_train_main, y_train_main,
            eval_set=eval_set,
            eval_metric=['logloss', 'error'],
            verbose=True
        )
        
        # Extract learning curves from the model
        results = model.evals_result()
        training_info['learning_curves']['train_loss'] = results['validation_0']['logloss']
        training_info['learning_curves']['val_loss'] = results['validation_1']['logloss']
        training_info['learning_curves']['train_accuracy'] = [1-x for x in results['validation_0']['error']]
        training_info['learning_curves']['val_accuracy'] = [1-x for x in results['validation_1']['error']]
        
        # Plot learning curves
        n_estimators = len(training_info['learning_curves']['train_loss'])
        x_axis = range(0, n_estimators)
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(x_axis, training_info['learning_curves']['train_loss'], label='Train')
        plt.plot(x_axis, training_info['learning_curves']['val_loss'], label='Validation')
        plt.legend()
        plt.ylabel('Log Loss')
        plt.title('XGBoost Loss')
        
        plt.subplot(1, 2, 2)
        plt.plot(x_axis, training_info['learning_curves']['train_accuracy'], label='Train')
        plt.plot(x_axis, training_info['learning_curves']['val_accuracy'], label='Validation')
        plt.legend()
        plt.ylabel('Accuracy')
        plt.title('XGBoost Accuracy')
        
        plt.tight_layout()
        print("Learning curves captured for XGBoost")
        
    else:
        # For other models, simply fit on balanced data
        model.fit(X_train_balanced, y_train_balanced)
    
    training_time = time() - start_time
    
    training_info['training_time'] = training_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    return model, training_info

def train_and_evaluate_model_balanced(dataset, model_type, balance_method='smote', tune_hyperparams=False, output_dir=None, use_gpu=False):
    """
    Train and evaluate a model of the given type on a balanced version of the dataset.
    
    Parameters:
    -----------
    dataset : dict
        Dataset dictionary
    model_type : str
        Type of model to train
    balance_method : str
        Method to use for balancing the dataset
    tune_hyperparams : bool
        Whether to perform hyperparameter tuning
    output_dir : str, optional
        Directory to save the model and results
    use_gpu : bool
        Whether to use GPU acceleration if available
        
    Returns:
    --------
    tuple
        (model, metrics, training_info)
    """
    # Create model without class weights since we'll balance the data
    print(f"Creating {MODEL_TYPES.get(model_type, model_type)} model")
    if use_gpu:
        print(f"GPU acceleration enabled for compatible models")
    model = create_model(model_type, class_weight=None, use_gpu=use_gpu)
    
    # Train model on balanced data
    model, training_info = train_model_balanced(model, dataset, tune_hyperparams, balance_method)
    
    # Evaluate model on unbalanced test data to get real-world performance
    metrics = evaluate_model(model, dataset, output_dir)
    
    # Save model and results if output_dir is specified
    if output_dir:
        model_name = f"{dataset['name']}_{model_type}_{balance_method}"
        save_model_and_results(model, dataset, metrics, training_info, output_dir, model_name)
    
    return model, metrics, training_info

def train_all_models_balanced(dataset, model_types=None, balance_method='smote', tune_hyperparams=False, output_dir=None, use_gpu=False):
    """
    Train all specified models on balanced versions of the dataset.
    
    Parameters:
    -----------
    dataset : dict
        Dataset dictionary
    model_types : list of str, optional
        Types of models to train (default: all supported types)
    balance_method : str
        Method to use for balancing the dataset
    tune_hyperparams : bool
        Whether to perform hyperparameter tuning
    output_dir : str, optional
        Directory to save the models and results
    use_gpu : bool
        Whether to use GPU acceleration if available
        
    Returns:
    --------
    dict
        Dictionary mapping model types to (model, metrics, training_info) tuples
    """
    if model_types is None:
        model_types = list(MODEL_TYPES.keys())
    
    results = {}
    
    for model_type in model_types:
        print(f"\n{'='*50}")
        print(f"Training {MODEL_TYPES.get(model_type, model_type)} on balanced {dataset['name']} dataset using {balance_method}")
        print(f"{'='*50}")
        
        # Create model-specific output directory if needed
        model_output_dir = None
        if output_dir:
            model_output_dir = os.path.join(output_dir, f"{dataset['name']}_{model_type}_{balance_method}")
            os.makedirs(model_output_dir, exist_ok=True)
        
        # Train and evaluate model
        try:
            model, metrics, training_info = train_and_evaluate_model_balanced(
                dataset, model_type, balance_method, tune_hyperparams, model_output_dir, use_gpu
            )
            results[model_type] = (model, metrics, training_info)
        except Exception as e:
            print(f"Error training {model_type} model: {e}")
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description='Train and evaluate multiple machine learning models on balanced datasets',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--dataset', type=str, default='contraction',
                      help='Dataset to use (contraction, selected, top50 or path to dataset directory)')
    parser.add_argument('--output_dir', type=str, default='results/balanced',
                      help='Directory to save trained models and results')
    parser.add_argument('--models', type=str, nargs='+', default=list(MODEL_TYPES.keys()),
                      choices=list(MODEL_TYPES.keys()),
                      help='Models to train')
    parser.add_argument('--balance_method', type=str, default='smote',
                      choices=['smote', 'random_over', 'random_under', 'smote_enn', 'smote_tomek'],
                      help='Method to use for balancing the dataset')
    parser.add_argument('--tune_hyperparams', action='store_true',
                      help='Perform hyperparameter tuning')
    parser.add_argument('--normalize', action='store_true',
                      help='Normalize features')
    parser.add_argument('--all_datasets', action='store_true',
                      help='Train on all available datasets')
    parser.add_argument('--use_gpu', action='store_true',
                      help='Use GPU acceleration for compatible models')
    
    args = parser.parse_args()
    
    # Print GPU info if requested
    if args.use_gpu:
        print("GPU acceleration requested")
        # This will trigger the GPU detection prints added in the imports section
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get list of datasets to use
    datasets_to_process = []
    if args.all_datasets:
        print("Training on all available datasets")
        datasets_to_process = list(DATASET_PATHS.keys())
    else:
        datasets_to_process = [args.dataset]
    
    # Process each dataset
    for dataset_name in datasets_to_process:
        print(f"\n{'#'*80}")
        print(f"Processing dataset: {dataset_name} with {args.balance_method} balancing")
        print(f"{'#'*80}")
        
        # Load dataset
        dataset = load_dataset(dataset_name)
        
        if dataset is None:
            print(f"Failed to load dataset '{dataset_name}'. Skipping.")
            continue
        
        # Preprocess dataset
        processed_dataset = preprocess_data(dataset, normalize=args.normalize)
        
        # Create dataset-specific output directory
        dataset_output_dir = os.path.join(args.output_dir, f"{dataset['name']}_{args.balance_method}")
        os.makedirs(dataset_output_dir, exist_ok=True)
        
        # Train all specified models
        results = train_all_models_balanced(
            processed_dataset, 
            model_types=args.models,
            balance_method=args.balance_method,
            tune_hyperparams=args.tune_hyperparams,
            output_dir=dataset_output_dir,
            use_gpu=args.use_gpu
        )
        
        # Save summary of all models for this dataset
        summary = {
            'dataset': dataset['name'],
            'balance_method': args.balance_method,
            'models': {}
        }
        
        for model_type, (_, metrics, training_info) in results.items():
            # Extract key metrics
            if 'classification_report' in metrics:
                report = metrics['classification_report']
                
                # Always use class 1.0 as the positive class for contraction detection
                pos_class = '1.0'
                
                # Ensure the positive class exists in the report
                if pos_class not in report:
                    print(f"Warning: Class '{pos_class}' not found in classification report.")
                    # Try to find an alternative class label
                    for class_label in report.keys():
                        if class_label not in ['accuracy', 'macro avg', 'weighted avg'] and class_label != '0.0':
                            pos_class = class_label
                            print(f"Using alternative class label '{pos_class}' as the positive class")
                            break
                
                print(f"Using class label '{pos_class}' as the positive class for metrics")
                
                summary['models'][model_type] = {
                    'accuracy': report.get('accuracy', 0),
                    'precision': report.get(pos_class, {}).get('precision', 0),
                    'recall': report.get(pos_class, {}).get('recall', 0),
                    'f1': report.get(pos_class, {}).get('f1-score', 0),
                    'training_time': training_info.get('training_time', 0),
                    'original_class_distribution': training_info.get('original_class_distribution', {}),
                    'balanced_class_distribution': training_info.get('balanced_class_distribution', {})
                }
            
            # Add clinical metrics if available
            if 'clinical_metrics' in metrics:
                summary['models'][model_type].update(metrics['clinical_metrics'])
            
            # Add ROC AUC if available
            if 'roc' in metrics:
                summary['models'][model_type]['auc'] = metrics['roc']['auc']
        
        # Save summary to file
        summary_path = os.path.join(dataset_output_dir, 'summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nSummary saved to {summary_path}")
        
        # Print summary table
        print("\nModel Comparison:")
        print("-" * 80)
        print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'AUC':<10}")
        print("-" * 80)
        
        for model_type, metrics_dict in summary['models'].items():
            model_name = MODEL_TYPES.get(model_type, model_type)
            print(f"{model_name:<20} {metrics_dict.get('accuracy', 0):<10.4f} "
                  f"{metrics_dict.get('precision', 0):<10.4f} {metrics_dict.get('recall', 0):<10.4f} "
                  f"{metrics_dict.get('f1', 0):<10.4f} {metrics_dict.get('auc', 0):<10.4f}")
        
        print("-" * 80)
    
    print("\nTraining and evaluation complete!")

if __name__ == "__main__":
    main()
