#!/usr/bin/env python
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
import argparse
from tqdm import tqdm
import pickle
import time
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import json
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.abspath('.'))
from prepare_contraction_data import load_contraction_annotations_from_csv, prepare_full_dataset
from load_processed import get_available_records
from preprocess_ehg import preprocess_record
from train_contraction_model import train_with_hyperparameter_tuning, preprocess_data, evaluate_model, save_model
from sklearn.metrics import roc_curve, auc

param_grid = {
    'first_stage_percentile': [65, 70, 75],
    'second_stage_multiplier': [1.15, 1.2, 1.25],
    'window_size': [30, 45, 60],
    'step_size': [3, 5, 10],
    'label_threshold': [0.3, 0.5, 0.7],
    'lowcut': [0.1, 0.2, 0.3],
    'highcut': [4.0, 5.0, 6.0],
    'target_fs': [20, 50, 100],
    'skip_artifact_removal': [True, False]
}

def eval(model, X_test, y_test, feature_names):
    # Make predictions
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    y_pred = model.predict(X_test)

    metrics = {}
    # ROC curve and AUC
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        metrics['roc'] = {'auc': roc_auc}

    # F1 score
    metrics['f1'] = f1_score(y_test, y_pred)

    # Feature importance
    if hasattr(model, 'feature_importances_') and feature_names is not None:
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        top_n = min(20, len(feature_names))
        
        metrics['feature_importance'] = {
            'feature_names': [feature_names[i] for i in indices[:top_n]],
            'importance': importances[indices[:top_n]].tolist()
        }
    
    return metrics

def run_hyperparameter_search(annotations_dict, n_trials=36, n_records=20, verbose=True):
    """
    Run hyperparameter search with random sampling.
    
    Parameters:
    -----------
    annotations_dict : dict
        Dictionary mapping record names to contraction annotations
    n_trials : int
        Number of trials to run
    n_records : int
        Number of records to sample each time
    """
    # Get list of available records
    std_records, _ = get_available_records()
    valid_records = [r for r in std_records if r in annotations_dict]
    
    # Initialize results storage
    results = []
    
    for trial in tqdm(range(14, 14+n_trials), desc="Running hyperparameter search"):
        # Randomly sample parameters from the grid
        params = {k: random.choice(v) for k, v in param_grid.items()}
        
        trial_dir = f'data/trial_{trial+1}'
        processed_dir = trial_dir + '/processed'
        dataset_dir = trial_dir + '/contraction_data'
        os.makedirs(trial_dir, exist_ok=True)
        os.makedirs(processed_dir, exist_ok=True)
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Save parameters to a file in the trial directory
        with open(os.path.join(trial_dir, 'parameters.json'), 'w') as f:
            json.dump(params, f, indent=4)
        
        # Randomly sample records
        sampled_records = random.sample(valid_records, n_records)
        if verbose:
            print(f"Sampled records:", sampled_records)
        
        # Process each record
        for record in sampled_records:
            # Process the signal with current parameters
            processed_signals, _, _, processing_info = preprocess_record(
                record_name=record,
                data_dir="data/records",
                output_dir=processed_dir,  # Use trial-specific directory
                lowcut=params['lowcut'],
                highcut=params['highcut'],
                target_fs=params['target_fs'],
                trim_seconds=60,  # Fixed value as it's not in our grid
                skip_artifact_removal=params['skip_artifact_removal'],
                save_intermediate=False,
                verbose=False
            )
            
            if processed_signals is None:
                print(f"Failed to process record {record}")
            continue
        

        if verbose:
            print(f"Processed {len(sampled_records)} records")
            
        # Extract features from processed signals
        _ = prepare_full_dataset(
            annotations_dict,
            processed_dir=processed_dir,
            output_dir=dataset_dir,
            window_size=params['window_size'],
            step_size=params['step_size'],
            label_threshold=params['label_threshold'],
            first_stage_percentile=params['first_stage_percentile'],
            second_stage_multiplier=params['second_stage_multiplier'],
            no_adjust_trim=params['skip_artifact_removal'],
            verbose=False
        )
        X = np.load(os.path.join(dataset_dir, 'X.npy'))
        y = np.load(os.path.join(dataset_dir, 'y.npy'))
        feature_names = np.load(os.path.join(dataset_dir, 'feature_names.npy'))
        print(len(feature_names))

        if verbose:
            print(f"Generated {len(X)} features")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Preprocess data
        X_train_proc, X_val_proc, preprocessing_info = preprocess_data(
            X_train, X_val, normalize=True
        )
        
        # Check class distribution
        train_pos_percentage = np.mean(y_train) * 100
        val_pos_percentage = np.mean(y_val) * 100

        random_forest_model= train_with_hyperparameter_tuning(X_train_proc, y_train, model_type='random_forest')
        random_forest_metrics = eval(random_forest_model, X_val_proc, y_val, feature_names)

        if verbose:
            print(f"Trained models")
        
        # Save the trained models
        save_model(random_forest_model, preprocessing_info, trial_dir, model_name="random_forest_model")

        if verbose:
            print(f"Trained models saved to {trial_dir}")
        
        # Store results
        trial_results = {
            'trial': trial + 1,
            'n_records': n_records,
            'n_samples': len(y),
            'positive_percentage': np.mean(y) * 100,
            'random_forest_metrics': random_forest_metrics
        }
        
        # Add all parameter values to results
        trial_results.update(params)

        # Save trial results in trial directory
        with open(os.path.join(trial_dir, 'trial_results.json'), 'w') as f:
            json.dump(trial_results, f, indent=4)
        
        results.append(trial_results)
        
        # Print progress
        print(f"\nTrial {trial+1} Results:")
        # print(f"Parameters: {params}")
        print(f"Samples: {len(y)} ({np.mean(y)*100:.2f}% positive)")
        print(f"Validation metrics: random_forest_auc={trial_results['random_forest_metrics']['roc']['auc']:.4f}, f1={trial_results['random_forest_metrics']['f1']:.4f}")
        print(f"Data and model saved to: {trial_dir}")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    results_df.to_csv(os.path.join(trial_dir, 'hyperparameter_search_results.csv'), index=False)
    print(f"\nResults saved to {trial_dir}")
    
    # Print summary
    print("\nHyperparameter Search Summary:")

    # Find best trial for Random Forest by F1 score
    best_rf_idx = results_df['random_forest_metrics'].apply(lambda m: m['f1']).idxmax()
    best_rf_trial = results_df.loc[best_rf_idx]
    best_rf_f1 = best_rf_trial['random_forest_metrics']['f1']

    print(f"Best Random Forest F1: {best_rf_f1:.4f}")

    print("\nBest Random Forest trial parameters:")
    for param in param_grid.keys():
        print(f"{param}: {best_rf_trial[param]}")
    print(f"Number of records: {best_rf_trial['n_records']}")
    print(f"Random Forest F1: {best_rf_f1:.4f}")
    print(f"Best trial : {best_rf_trial['trial']}")

    # Create visualizations for F1 score
    rf_f1s = results_df['random_forest_metrics'].apply(lambda m: m['f1'] if 'f1' in m else np.nan)

    plt.figure(figsize=(12, 8))
    plt.scatter(results_df['window_size'], rf_f1s, c=results_df['first_stage_percentile'], cmap='viridis', label='Random Forest F1')
    plt.colorbar(label='First Stage Percentile')
    plt.xlabel('Window Size (seconds)')
    plt.ylabel('Validation F1 Score')
    plt.title('Hyperparameter Search Results (F1 Score)')
    plt.legend()
    plt.savefig(os.path.join(trial_dir, 'hyperparameter_search_f1.png'), dpi=300, bbox_inches='tight')
    
    return results_df



def main():
    parser = argparse.ArgumentParser(
        description='Run hyperparameter search for contraction detection',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--csv_path', type=str, default='contractions.csv',
                      help='Path to the CSV file with contraction annotations')
    parser.add_argument('--processed_dir', type=str, default='data/processed',
                      help='Directory containing processed signals')
    parser.add_argument('--n_trials', type=int, default=50,
                      help='Number of trials to run')
    parser.add_argument('--n_records', type=int, default=15,
                      help='Maximum number of records to sample')
    parser.add_argument('--verbose', type=bool, default=True,
                      help='Verbose output')
    
    args = parser.parse_args()
    
    # Load annotations
    annotations_dict = load_contraction_annotations_from_csv(args.csv_path)
    print(f"Loaded annotations for {len(annotations_dict)} records")
    
    # Run hyperparameter search
    results = run_hyperparameter_search(
        annotations_dict,
        n_trials=args.n_trials,
        n_records=args.n_records
    )


if __name__ == "__main__":
    main() 