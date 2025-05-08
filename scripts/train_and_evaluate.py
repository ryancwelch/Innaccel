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

# Ignore warnings
warnings.filterwarnings('ignore')

# GPU support imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging
try:
    import tensorflow as tf
    tf_available = True
    # Check for GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"TensorFlow detected {len(gpus)} GPU(s)")
        for gpu in gpus:
            print(f"  Name: {gpu.name}, Type: {gpu.device_type}")
        # Allow memory growth to avoid taking all GPU memory
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("No GPUs detected by TensorFlow")
except ImportError:
    tf_available = False
    print("TensorFlow not available")

try:
    import torch
    torch_available = True
    # Check for GPU availability
    if torch.cuda.is_available():
        cuda_version = torch.version.cuda
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
        print(f"PyTorch detected {device_count} GPU(s)")
        print(f"  CUDA version: {cuda_version}")
        print(f"  Device name: {device_name}")
    else:
        print("No GPUs detected by PyTorch")
except ImportError:
    torch_available = False
    print("PyTorch not available")

# Supported models
MODEL_TYPES = {
    'rf': 'Random Forest',
    'xgb': 'XGBoost',
    'lr': 'Logistic Regression',
    'svm': 'Support Vector Machine',
    'mlp': 'Neural Network (MLP)'
}

# Define dataset paths
DATASET_PATHS = {
    'non_sequential_contraction': 'data/mit-innaccel/non_sequential_contraction_data',
    'non_sequential_top50': 'data/mit-innaccel/non_sequential_top50_selected_features',
    'non_sequential_top600': 'data/mit-innaccel/non_sequential_top600_selected_features'
}

def load_dataset(dataset_name, data_dir=None):
    """
    Load a dataset by name or path.
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset or path to the dataset directory
    data_dir : str, optional
        Base directory for datasets
        
    Returns:
    --------
    dict
        Dictionary containing dataset components
    """
    try:
        # Try to use predefined dataset path
        if dataset_name in DATASET_PATHS:
            data_path = DATASET_PATHS[dataset_name]
        else:
            # Otherwise use provided path
            data_path = dataset_name
            
        # If data_dir is provided, join with data_path
        if data_dir:
            data_path = os.path.join(data_dir, data_path)
            
        print(f"Loading dataset from {data_path}")
        
        # Load data
        X_train = np.load(os.path.join(data_path, 'X_train.npy'))
        X_test = np.load(os.path.join(data_path, 'X_test.npy'))
        y_train = np.load(os.path.join(data_path, 'y_train.npy'))
        y_test = np.load(os.path.join(data_path, 'y_test.npy'))
        
        # Load feature names if available
        try:
            feature_names = np.load(os.path.join(data_path, 'feature_names.npy'), allow_pickle=True)
        except FileNotFoundError:
            feature_names = None
            print("Feature names not found")
            
        # Load dataset info if available
        try:
            dataset_info = np.load(os.path.join(data_path, 'dataset_info.npy'), allow_pickle=True).item()
        except FileNotFoundError:
            dataset_info = None
            print("Dataset info not found")

        print(f"Dataset loaded: X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
        print(f"Class distribution - Train: {np.bincount(y_train.astype(int))}, Test: {np.bincount(y_test.astype(int))}")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': feature_names,
            'dataset_info': dataset_info,
            'name': os.path.basename(data_path)
        }
        
    except Exception as e:
        print(f"Error loading dataset '{dataset_name}': {e}")
        return None

def preprocess_data(dataset, normalize=True):
    """
    Preprocess the dataset.
    
    Parameters:
    -----------
    dataset : dict
        Dataset dictionary from load_dataset
    normalize : bool
        Whether to normalize features
        
    Returns:
    --------
    dict
        Processed dataset with additional preprocessing info
    """
    X_train = dataset['X_train']
    X_test = dataset['X_test']
    
    # Handle missing values
    X_train_proc = np.nan_to_num(X_train)
    X_test_proc = np.nan_to_num(X_test)
    
    # Store preprocessing info
    preprocessing_info = {}
    
    # Normalize if requested
    if normalize:
        scaler = StandardScaler()
        X_train_proc = scaler.fit_transform(X_train_proc)
        X_test_proc = scaler.transform(X_test_proc)
        preprocessing_info['scaler'] = scaler
    
    preprocessing_info['normalize'] = normalize
    
    # Create new dataset with processed data
    processed_dataset = dataset.copy()
    processed_dataset['X_train'] = X_train_proc
    processed_dataset['X_test'] = X_test_proc
    processed_dataset['preprocessing_info'] = preprocessing_info
    
    return processed_dataset

def create_model(model_type, class_weight=None, random_state=42, use_gpu=False):
    """
    Create a model instance based on the model type.
    
    Parameters:
    -----------
    model_type : str
        Type of model to create
    class_weight : str or dict, optional
        Class weights for imbalanced datasets
    random_state : int
        Random seed for reproducibility
    use_gpu : bool
        Whether to use GPU acceleration if available
        
    Returns:
    --------
    model
        Instantiated model
    """
    if model_type == 'rf':
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=random_state,
            class_weight=class_weight,
            n_jobs=-1
        )
    
    elif model_type == 'xgb':
        # Configure XGBoost for GPU if available
        if use_gpu and 'XGBoost_USE_CUDA' in os.environ:
            return xgb.XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state,
                n_jobs=-1,
                tree_method='gpu_hist',  # Use GPU acceleration
                gpu_id=0,                # Use first GPU
                predictor='gpu_predictor',  # Use GPU for prediction
            )
        else:
            return xgb.XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state,
                n_jobs=-1
            )
    
    elif model_type == 'lr':
        return LogisticRegression(
            penalty='l2',
            C=1.0,
            max_iter=1000,
            random_state=random_state,
            class_weight=class_weight,
            n_jobs=-1
        )
    
    elif model_type == 'svm':
        return SVC(
            C=1.0,
            kernel='rbf',
            probability=True,
            random_state=random_state,
            class_weight=class_weight
        )
    
    elif model_type == 'mlp':
        if use_gpu and tf_available:
            # Define a TensorFlow-based MLP for GPU acceleration
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense, Dropout
            from tensorflow.keras.optimizers import Adam
            from tensorflow.keras.callbacks import EarlyStopping
            from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
            
            # Define a function to create the MLP model
            def create_keras_mlp():
                model = Sequential([
                    Dense(100, activation='relu', input_shape=(None,)),
                    Dropout(0.2),
                    Dense(50, activation='relu'),
                    Dropout(0.2),
                    Dense(1, activation='sigmoid')
                ])
                model.compile(
                    optimizer=Adam(learning_rate=0.001),
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
                return model
            
            return KerasClassifier(
                build_fn=create_keras_mlp,
                epochs=100,
                batch_size=32,
                verbose=1,
                callbacks=[EarlyStopping(monitor='val_loss', patience=10)],
                validation_split=0.1,
                random_state=random_state
            )
        else:
            # Fall back to sklearn's MLP implementation
            return MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                batch_size='auto',
                learning_rate='adaptive',
                max_iter=500,
                random_state=random_state
            )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def train_model(model, dataset, tune_hyperparams=False):
    """
    Train a model on the dataset.
    
    Parameters:
    -----------
    model : sklearn model
        Model to train
    dataset : dict
        Processed dataset dictionary
    tune_hyperparams : bool
        Whether to perform hyperparameter tuning
        
    Returns:
    --------
    model
        Trained model
    dict
        Training information
    """
    X_train = dataset['X_train']
    y_train = dataset['y_train']
    
    # Create a validation set
    X_train_main, X_val, y_train_main, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    training_info = {
        'training_time': None,
        'best_params': None,
        'learning_curves': {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': []
        }
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
            grid_search.fit(X_train, y_train)
            training_time = time() - start_time
            
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best score: {grid_search.best_score_:.4f}")
            
            model = grid_search.best_estimator_
            training_info['best_params'] = grid_search.best_params_
            training_info['training_time'] = training_time
            
            return model, training_info
    
    # Regular training without hyperparameter tuning
    print(f"Training model...")
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
        
    elif isinstance(model, MLPClassifier):
        # For scikit-learn MLPClassifier, can access loss_curve_ after training
        model.fit(X_train_main, y_train_main)
        
        # Store training loss
        training_info['learning_curves']['train_loss'] = model.loss_curve_
        
        # Calculate validation loss at each epoch
        val_losses = []
        val_accuracies = []
        train_accuracies = []
        
        # Clone the model at each iteration point
        base_model = clone(model)
        base_model.max_iter = 1  # Set to 1 to increment manually
        base_model.warm_start = True  # Use warm start to continue training
        
        # Train incrementally to track validation performance
        for i in range(1, model.n_iter_ + 1):
            base_model.max_iter = i
            base_model.fit(X_train_main, y_train_main)
            
            # Validation scores
            y_val_pred = base_model.predict(X_val)
            val_accuracies.append(accuracy_score(y_val, y_val_pred))
            
            # Training scores
            y_train_pred = base_model.predict(X_train_main)
            train_accuracies.append(accuracy_score(y_train_main, y_train_pred))
        
        training_info['learning_curves']['val_accuracy'] = val_accuracies
        training_info['learning_curves']['train_accuracy'] = train_accuracies
        
        # Plot learning curves
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(range(len(model.loss_curve_)), model.loss_curve_, label='Train Loss')
        plt.legend()
        plt.ylabel('Loss')
        plt.title('MLP Loss Curve')
        
        plt.subplot(1, 2, 2)
        plt.plot(range(len(train_accuracies)), train_accuracies, label='Train Accuracy')
        plt.plot(range(len(val_accuracies)), val_accuracies, label='Validation Accuracy')
        plt.legend()
        plt.ylabel('Accuracy')
        plt.title('MLP Accuracy Curve')
        
        plt.tight_layout()
        print("Learning curves captured for MLP")
        
    elif isinstance(model, RandomForestClassifier):
        # For Random Forest, track performance as trees are added
        # This is an approximation since RF doesn't train sequentially
        model.fit(X_train_main, y_train_main)
        
        # Create array to store performance as trees are added
        n_estimators = model.n_estimators
        train_scores = []
        val_scores = []
        
        # For each subset of trees, calculate performance
        for i in range(1, n_estimators + 1, max(1, n_estimators // 10)):
            # Create a new RF with fewer trees
            subset_model = RandomForestClassifier(
                n_estimators=i,
                max_depth=model.max_depth,
                min_samples_split=model.min_samples_split,
                min_samples_leaf=model.min_samples_leaf,
                random_state=model.random_state,
                class_weight=model.class_weight,
                n_jobs=model.n_jobs
            )
            subset_model.fit(X_train_main, y_train_main)
            
            # Calculate scores
            train_pred = subset_model.predict(X_train_main)
            val_pred = subset_model.predict(X_val)
            
            train_score = accuracy_score(y_train_main, train_pred)
            val_score = accuracy_score(y_val, val_pred)
            
            train_scores.append(train_score)
            val_scores.append(val_score)
        
        training_info['learning_curves']['train_accuracy'] = train_scores
        training_info['learning_curves']['val_accuracy'] = val_scores
        
        # Plot learning curve
        plt.figure(figsize=(8, 6))
        tree_counts = list(range(1, n_estimators + 1, max(1, n_estimators // 10)))
        plt.plot(tree_counts, train_scores, label='Train Accuracy')
        plt.plot(tree_counts, val_scores, label='Validation Accuracy')
        plt.xlabel('Number of Trees')
        plt.ylabel('Accuracy')
        plt.title('Random Forest Learning Curve')
        plt.legend()
        plt.tight_layout()
        print("Learning curves captured for Random Forest")
        
    else:
        # For other models, use cross-validation to track performance
        model.fit(X_train, y_train)
        print("No learning curves available for this model type")
    
    training_time = time() - start_time
    
    training_info['training_time'] = training_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    return model, training_info

def evaluate_model(model, dataset, output_dir=None):
    """
    Evaluate a trained model on test data.
    
    Parameters:
    -----------
    model : sklearn model
        Trained model
    dataset : dict
        Dataset dictionary
    output_dir : str, optional
        Directory to save evaluation results
        
    Returns:
    --------
    dict
        Evaluation metrics
    """
    X_test = dataset['X_test']
    y_test = dataset['y_test']
    feature_names = dataset.get('feature_names')
    
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
    metrics['confusion_matrix'] = conf_matrix.tolist()
    
    # Print metrics
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(conf_matrix)
    
    # Clinical metrics
    tn, fp, fn, tp = conf_matrix.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    metrics['clinical_metrics'] = {
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'ppv': float(ppv),
        'npv': float(npv)
    }
    
    print("\nClinical Metrics:")
    print(f"Sensitivity (Recall): {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Positive Predictive Value (Precision): {ppv:.4f}")
    print(f"Negative Predictive Value: {npv:.4f}")
    
    # Only create plots if output_dir is specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
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
        
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # ROC curve and AUC
        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            
            metrics['roc'] = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'auc': float(roc_auc)
            }
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            
            plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # Feature importance
        if hasattr(model, 'feature_importances_') and feature_names is not None:
            try:
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]
                
                # Make sure we don't have more indices than feature_names
                if len(indices) > len(feature_names):
                    print(f"Warning: Model has {len(indices)} features but only {len(feature_names)} feature names provided.")
                    # Create generic feature names for any missing ones
                    extended_feature_names = np.array(list(feature_names) + 
                                                  [f'feature_{i}' for i in range(len(feature_names), len(indices))])
                else:
                    extended_feature_names = np.array(feature_names)
                
                top_n = min(20, len(indices))
                
                metrics['feature_importance'] = {
                    'feature_names': [str(extended_feature_names[i]) for i in indices[:top_n]],
                    'importance': importances[indices[:top_n]].tolist()
                }
                
                plt.figure(figsize=(12, 8))
                plt.title("Feature Importances (Top 20)")
                plt.bar(range(top_n), importances[indices[:top_n]], align="center")
                plt.xticks(range(top_n), [extended_feature_names[i] for i in indices[:top_n]], rotation=90)
                plt.tight_layout()
                
                plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
                plt.close()
            except Exception as e:
                print(f"Error plotting feature importance: {e}")
                print("Skipping feature importance plot.")
    
    return metrics

def save_model_and_results(model, dataset, metrics, training_info, output_dir, model_name):
    """
    Save model, preprocessing information, and evaluation metrics.
    
    Parameters:
    -----------
    model : sklearn model
        Trained model
    dataset : dict
        Dataset dictionary
    metrics : dict
        Evaluation metrics
    training_info : dict
        Training information
    output_dir : str
        Directory to save the model and results
    model_name : str
        Name for the saved model files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, f"{model_name}.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    # Save preprocessing information
    preprocessing_path = os.path.join(output_dir, f"{model_name}_preprocessing.joblib")
    preprocessing_info = {
        'normalize': dataset.get('normalize', False),
        'feature_names': dataset.get('feature_names', None),
        'feature_stats': dataset.get('feature_stats', None)
    }
    joblib.dump(preprocessing_info, preprocessing_path)
    print(f"Preprocessing info saved to {preprocessing_path}")
    
    # Save metrics
    metrics_path = os.path.join(output_dir, f"{model_name}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")
    
    # Save training info
    training_info_path = os.path.join(output_dir, f"{model_name}_training_info.json")
    
    # Convert numpy arrays to lists for JSON serialization
    for key in training_info.get('learning_curves', {}):
        if isinstance(training_info['learning_curves'][key], np.ndarray):
            training_info['learning_curves'][key] = training_info['learning_curves'][key].tolist()
        elif isinstance(training_info['learning_curves'][key], list):
            for i, val in enumerate(training_info['learning_curves'][key]):
                if isinstance(val, np.ndarray) or isinstance(val, np.number):
                    training_info['learning_curves'][key][i] = float(val)
    
    with open(training_info_path, 'w') as f:
        json.dump(training_info, f, indent=2)
    print(f"Training info saved to {training_info_path}")
    
    # Save learning curve plots if available
    if 'learning_curves' in training_info and any(training_info['learning_curves'].values()):
        learning_curves_path = os.path.join(output_dir, 'learning_curves.png')
        plt.savefig(learning_curves_path)
        plt.close()
        print(f"Learning curves saved to {learning_curves_path}")

def train_and_evaluate_model(dataset, model_type, tune_hyperparams=False, output_dir=None, use_gpu=False):
    """
    Train and evaluate a model of the given type on the dataset.
    
    Parameters:
    -----------
    dataset : dict
        Dataset dictionary
    model_type : str
        Type of model to train
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
    # Create class weights based on class distribution
    y_train = dataset['y_train']
    class_counts = np.bincount(y_train.astype(int))
    total_samples = len(y_train)
    class_weight = {i: total_samples / (len(class_counts) * count) for i, count in enumerate(class_counts)}
    
    # Create model
    print(f"Creating {MODEL_TYPES.get(model_type, model_type)} model")
    if use_gpu:
        print(f"GPU acceleration enabled for compatible models")
    model = create_model(model_type, class_weight=class_weight, use_gpu=use_gpu)
    
    # Train model
    model, training_info = train_model(model, dataset, tune_hyperparams)
    
    # Evaluate model
    metrics = evaluate_model(model, dataset, output_dir)
    
    # Save model and results if output_dir is specified
    if output_dir:
        model_name = f"{dataset['name']}_{model_type}"
        save_model_and_results(model, dataset, metrics, training_info, output_dir, model_name)
    
    return model, metrics, training_info

def train_all_models(dataset, model_types=None, tune_hyperparams=False, output_dir=None, use_gpu=False):
    """
    Train all specified models on the dataset.
    
    Parameters:
    -----------
    dataset : dict
        Dataset dictionary
    model_types : list of str, optional
        Types of models to train (default: all supported types)
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
        print(f"Training {MODEL_TYPES.get(model_type, model_type)} on {dataset['name']} dataset")
        print(f"{'='*50}")
        
        # Create model-specific output directory if needed
        model_output_dir = None
        if output_dir:
            model_output_dir = os.path.join(output_dir, f"{dataset['name']}_{model_type}")
            os.makedirs(model_output_dir, exist_ok=True)
        
        # Train and evaluate model
        try:
            model, metrics, training_info = train_and_evaluate_model(
                dataset, model_type, tune_hyperparams, model_output_dir, use_gpu
            )
            results[model_type] = (model, metrics, training_info)
        except Exception as e:
            print(f"Error training {model_type} model: {e}")
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description='Train and evaluate multiple machine learning models on datasets',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--dataset', type=str, default='contraction',
                      help='Dataset to use (contraction, selected, top50 or path to dataset directory)')
    parser.add_argument('--output_dir', type=str, default='results',
                      help='Directory to save trained models and results')
    parser.add_argument('--models', type=str, nargs='+', default=list(MODEL_TYPES.keys()),
                      choices=list(MODEL_TYPES.keys()),
                      help='Models to train')
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
        print(f"Processing dataset: {dataset_name}")
        print(f"{'#'*80}")
        
        # Load dataset
        dataset = load_dataset(dataset_name)
        
        if dataset is None:
            print(f"Failed to load dataset '{dataset_name}'. Skipping.")
            continue
        
        # Preprocess dataset
        processed_dataset = preprocess_data(dataset, normalize=args.normalize)
        
        # Create dataset-specific output directory
        dataset_output_dir = os.path.join(args.output_dir, dataset['name'])
        os.makedirs(dataset_output_dir, exist_ok=True)
        
        # Train all specified models
        results = train_all_models(
            processed_dataset, 
            model_types=args.models,
            tune_hyperparams=args.tune_hyperparams,
            output_dir=dataset_output_dir,
            use_gpu=args.use_gpu
        )
        
        # Save summary of all models for this dataset
        summary = {
            'dataset': dataset['name'],
            'models': {}
        }
        
        for model_type, (_, metrics, training_info) in results.items():
            # Extract key metrics
            if 'classification_report' in metrics:
                report = metrics['classification_report']
                
                # Determine which class label is the positive class (could be '1' or 1 depending on the implementation)
                pos_class = None
                for class_label in report.keys():
                    if class_label not in ['accuracy', 'macro avg', 'weighted avg']:
                        # Try to convert to int to find the largest class label (positive class)
                        try:
                            if pos_class is None or int(class_label) > int(pos_class):
                                pos_class = class_label
                        except (ValueError, TypeError):
                            # Skip if not convertible to int
                            continue
                
                # If no positive class found, default to '1'
                if pos_class is None:
                    pos_class = '1'
                
                print(f"Using class label '{pos_class}' as the positive class for metrics")
                
                summary['models'][model_type] = {
                    'accuracy': report.get('accuracy', 0),
                    'precision': report.get(pos_class, {}).get('precision', 0),
                    'recall': report.get(pos_class, {}).get('recall', 0),
                    'f1': report.get(pos_class, {}).get('f1-score', 0),
                    'training_time': training_info.get('training_time', 0)
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