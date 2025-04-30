#!/usr/bin/env python
import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import glob
import joblib
import argparse
from train_contraction_model import load_dataset, train_with_hyperparameter_tuning, evaluate_model

def load_feature_names(trial_dir):
    feature_names_path = os.path.join(trial_dir, 'contraction_data', 'feature_names.npy')
    if os.path.exists(feature_names_path):
        return np.load(feature_names_path)
    return None

def load_feature_matrix_and_labels(trial_dir):
    X_path = os.path.join(trial_dir, 'contraction_data', 'X.npy')
    y_path = os.path.join(trial_dir, 'contraction_data', 'y.npy')
    if os.path.exists(X_path) and os.path.exists(y_path):
        return np.load(X_path), np.load(y_path)
    return None, None

def load_trial_results(trial_dir):
    results_path = os.path.join(trial_dir, 'trial_results.json')
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            return json.load(f)
    return None

def load_combined_trials():
    """Load all trial results from combined_trials.json"""
    with open('combined_trials.json', 'r') as f:
        return json.load(f)

def agg_trial_feature_importances():
    combined_trials = load_combined_trials()
    aggregated_rf_importance = {}
    total_f1 = 0
    f1_scores = []

    # First pass: collect F1 scores
    for trial_id, trial_data in combined_trials.items():
        trial_results = trial_data['trial_results.json']
        if 'random_forest_metrics' in trial_results:
            if 'f1' in trial_results['random_forest_metrics']:
                f1 = trial_results['random_forest_metrics']['f1']
                f1_scores.append(f1)
                total_f1 += f1

    # Second pass: collect all importance values for each feature
    feature_importance_values = {}  # Store all importance values for each feature
    for trial_id, trial_data in combined_trials.items():
        trial_results = trial_data['trial_results.json']
        if 'random_forest_metrics' in trial_results and 'feature_importance' in trial_results['random_forest_metrics']:
            rf_importance = trial_results['random_forest_metrics']['feature_importance']
            trial_f1 = trial_results['random_forest_metrics']['f1']
            weight = trial_f1 / total_f1 if total_f1 > 0 else 1.0 / len(combined_trials)
            
            for name, imp in zip(rf_importance['feature_names'], rf_importance['importance']):
                weighted_imp = imp * weight
                if name in feature_importance_values:
                    feature_importance_values[name].append(weighted_imp)
                else:
                    feature_importance_values[name] = [weighted_imp]

    # Calculate mean, std, and confidence intervals for each feature
    feature_stats = {}
    for name, values in feature_importance_values.items():
        mean = np.mean(values)
        std = np.std(values)
        # 95% confidence interval
        ci_lower = mean - 1.96 * std / np.sqrt(len(values))
        ci_upper = mean + 1.96 * std / np.sqrt(len(values))
        feature_stats[name] = {
            'mean': mean,
            'std': std,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }

    # Sort by mean importance
    sorted_features = sorted(feature_stats.items(), key=lambda x: x[1]['mean'], reverse=True)
    top_n = min(20, len(sorted_features))
    sorted_features = sorted_features[:top_n]

    results = {
        'feature_names': [name for name, _ in sorted_features],
        'importance': [stats['mean'] for _, stats in sorted_features],
        'std': [stats['std'] for _, stats in sorted_features],
        'ci_lower': [stats['ci_lower'] for _, stats in sorted_features],
        'ci_upper': [stats['ci_upper'] for _, stats in sorted_features],
        'avg_f1_score': np.mean(f1_scores) if f1_scores else 0
    }
    return results

def compute_combined_feature_importance(trial_importance, rf_importances, feature_names):
    """
    Compute combined feature importance from trial results and random forest.
    """
    # Create dictionaries to map feature names to their importance values
    trial_dict = dict(zip(trial_importance['feature_names'], trial_importance['importance']))
    rf_dict = dict(zip(feature_names, rf_importances))

    # Find the intersection of features
    common_features = set(trial_dict.keys()) & set(rf_dict.keys())

    # Compute combined importance for common features
    combined_importance = {}
    for feature in common_features:
        # Normalize each importance score to [0,1] range
        trial_norm = trial_dict[feature] / max(trial_dict.values())
        rf_norm = rf_dict[feature] / max(rf_dict.values())
        
        # Average the normalized importance scores
        combined_importance[feature] = (trial_norm + rf_norm) / 2

    # Sort by combined importance
    sorted_combined = sorted(combined_importance.items(), key=lambda x: x[1], reverse=True)

    # Compute correlation between importance values
    common_features_list = list(common_features)
    trial_values = [trial_dict[feature] for feature in common_features_list]
    rf_values = [rf_dict[feature] for feature in common_features_list]
    
    # Calculate correlation
    correlation = np.corrcoef(trial_values, rf_values)[0, 1]

    return {
        'common_features': [feature for feature, _ in sorted_combined],
        'combined_importance': [importance for _, importance in sorted_combined],
        'correlation': correlation
    }

def plot_feature_importance(importance_data, title, output_file):
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    # Create a DataFrame for easier plotting
    df = pd.DataFrame({
        'Feature': importance_data['feature_names'],
        'Importance': importance_data['importance'],
        'CI Lower': importance_data['ci_lower'],
        'CI Upper': importance_data['ci_upper']
    })
    
    # Create the bar plot
    ax = sns.barplot(x='Importance', y='Feature', data=df, palette='viridis')
    
    # Add error bars for confidence intervals
    xerr = np.array([df['Importance'] - df['CI Lower'], df['CI Upper'] - df['Importance']])
    ax.errorbar(x=df['Importance'], y=range(len(df)), xerr=xerr, 
                fmt='none', color='black', capsize=3)
    
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel('Feature Importance', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def plot_combined_importance(combined_data, output_file):
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    # Create a DataFrame for easier plotting
    df = pd.DataFrame({
        'Feature': combined_data['common_features'],
        'Combined Importance': combined_data['combined_importance'],
        'CI Lower': combined_data['ci_lower'],
        'CI Upper': combined_data['ci_upper']
    })
    
    # Create the bar plot
    ax = sns.barplot(x='Combined Importance', y='Feature', data=df, palette='viridis')
    
    # Add error bars for confidence intervals
    xerr = np.array([df['Combined Importance'] - df['CI Lower'], df['CI Upper'] - df['Combined Importance']])
    ax.errorbar(x=df['Combined Importance'], y=range(len(df)), xerr=xerr, 
                fmt='none', color='black', capsize=3)
    
    plt.title('Combined Feature Importance', fontsize=14, pad=20)
    plt.xlabel('Combined Importance Score', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    
    # Add correlation text
    plt.text(0.95, 0.05, f'Correlation: {combined_data["correlation"]:.3f}',
             transform=ax.transAxes, ha='right', va='bottom',
             bbox=dict(facecolor='white', alpha=0.8))
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def select_important_features(importance_data, threshold=0.95):
    """
    Select features that explain a certain percentage of the total importance.
    Args:
        importance_data: Dictionary containing feature names and importance values
        threshold: Percentage of total importance to explain (default: 0.95)
    Returns:
        Dictionary containing selected features and their statistics
    """
    # Calculate total importance
    total_importance = sum(importance_data['importance'])
    
    # Sort features by importance
    sorted_features = sorted(zip(importance_data['feature_names'], 
                               importance_data['importance'],
                               importance_data['ci_lower'],
                               importance_data['ci_upper']),
                           key=lambda x: x[1], reverse=True)
    
    # Calculate cumulative importance
    cumulative_importance = 0
    selected_features = []
    for name, imp, ci_lower, ci_upper in sorted_features:
        cumulative_importance += imp
        selected_features.append({
            'name': name,
            'importance': imp,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'cumulative_importance': cumulative_importance,
            'percentage_of_total': cumulative_importance / total_importance
        })
        if cumulative_importance / total_importance >= threshold:
            break
    
    return {
        'selected_features': selected_features,
        'total_importance': total_importance,
        'threshold': threshold,
        'num_features': len(selected_features)
    }

def plot_cumulative_importance(selected_features_data, output_file):
    """
    Plot the cumulative importance of selected features.
    """
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    # Create DataFrame for plotting
    df = pd.DataFrame(selected_features_data['selected_features'])
    
    # Create the line plot
    plt.plot(range(1, len(df) + 1), df['percentage_of_total'] * 100, 
             marker='o', linestyle='-', linewidth=2)
    
    # Add confidence intervals
    plt.fill_between(range(1, len(df) + 1),
                    (df['ci_lower'] / selected_features_data['total_importance']) * 100,
                    (df['ci_upper'] / selected_features_data['total_importance']) * 100,
                    alpha=0.2)
    
    # Add threshold line
    plt.axhline(y=selected_features_data['threshold'] * 100, 
                color='r', linestyle='--', 
                label=f'Threshold ({selected_features_data["threshold"]*100:.0f}%)')
    
    # Add feature names as x-axis labels
    plt.xticks(range(1, len(df) + 1), df['name'], rotation=45, ha='right')
    
    plt.title('Cumulative Feature Importance', fontsize=14, pad=20)
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Cumulative Importance (%)', fontsize=12)
    plt.legend()
    plt.grid(True)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def train_and_compare_models(output_dir):
    """
    Train and compare models using all features vs selected features.
    First trains models to get feature importances,
    then uses those to select features for second models.
    """
    # Load the dataset
    X_train, X_test, y_train, y_test, feature_names, _ = load_dataset(data_dir="data/final_contraction_data")
    
    # Train initial models to get feature importances
    print("\nTraining initial models to get feature importances...")
    
    # Random Forest
    print("\nTraining Random Forest model...")
    rf_model = train_with_hyperparameter_tuning(X_train, y_train, model_type='random_forest')
    rf_metrics = evaluate_model(rf_model, X_test, y_test, feature_names)
    rf_importances = rf_model.feature_importances_
    
    # Save the final Random Forest model
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, 'final_random_forest_model.joblib')
    joblib.dump(rf_model, model_path)
    print(f"\nSaved final Random Forest model to {model_path}")
    
    # Get feature indices for random forest
    rf_indices = np.argsort(rf_importances)[::-1]
    
    # Calculate cumulative importance
    rf_total = sum(rf_importances)
    rf_cumulative = 0
    rf_selected_indices = []
    
    for idx in rf_indices:
        rf_cumulative += rf_importances[idx]
        rf_selected_indices.append(idx)
        if rf_cumulative / rf_total >= 0.95:
            break
            
    rf_selected_features = [feature_names[i] for i in rf_selected_indices]
    
    print(f"\nSelected {len(rf_selected_indices)} features based on random forest importance")
    
    # Train model with selected features
    print("\nTraining model with selected features...")
    
    # Random Forest with selected features
    X_train_rf = X_train[:, rf_selected_indices]
    X_test_rf = X_test[:, rf_selected_indices]
    rf_selected_model = train_with_hyperparameter_tuning(X_train_rf, y_train, model_type='random_forest')
    rf_selected_metrics = evaluate_model(rf_selected_model, X_test_rf, y_test, rf_selected_features)
    
    # Save the selected features model
    selected_model_path = os.path.join(output_dir, 'selected_features_random_forest_model.joblib')
    joblib.dump(rf_selected_model, selected_model_path)
    print(f"\nSaved selected features Random Forest model to {selected_model_path}")
    
    # Compare metrics
    comparison = {
        'random_forest': {
            'full_model': {
                'f1': rf_metrics['classification_report']['weighted avg']['f1-score'],
                'auc': rf_metrics['roc']['auc'] if 'roc' in rf_metrics else None
            },
            'selected_model': {
                'f1': rf_selected_metrics['classification_report']['weighted avg']['f1-score'],
                'auc': rf_selected_metrics['roc']['auc'] if 'roc' in rf_selected_metrics else None
            },
            'num_features': {
                'full': X_train.shape[1],
                'selected': len(rf_selected_indices)
            },
            'selected_features': rf_selected_features,
            'feature_importances': {name: float(imp) for name, imp in zip(feature_names, rf_importances)}
        }
    }
    
    # Plot comparisons
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    metrics = ['F1 Score', 'AUC']
    full_values = [
        comparison['random_forest']['full_model']['f1'],
        comparison['random_forest']['full_model']['auc']
    ]
    selected_values = [
        comparison['random_forest']['selected_model']['f1'],
        comparison['random_forest']['selected_model']['auc']
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, full_values, width, label='All Features', color='skyblue')
    plt.bar(x + width/2, selected_values, width, label='Selected Features', color='lightgreen')
    
    plt.ylabel('Score')
    plt.title('Random Forest Model Performance Comparison')
    plt.xticks(x, metrics)
    plt.legend()
    
    # Add value labels
    for i, v in enumerate(full_values):
        plt.text(i - width/2, v + 0.01, f'{v:.3f}', ha='center')
    for i, v in enumerate(selected_values):
        plt.text(i + width/2, v + 0.01, f'{v:.3f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'random_forest_comparison.png'))
    plt.close()
    
    return comparison, rf_importances

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze feature importance from trial data')
    parser.add_argument('--hyperparameter-only', action='store_true',
                      help='Only analyze feature importance from hyperparameter search')
    parser.add_argument('--importance-threshold', type=float, default=0.95,
                      help='Threshold for cumulative feature importance (default: 0.95)')
    parser.add_argument('--compare-models', action='store_true',
                      help='Train and compare models using selected features')
    args = parser.parse_args()

    # Create analysis_results directory if it doesn't exist
    output_dir = 'analysis_results'
    os.makedirs(output_dir, exist_ok=True)

    results = {}
    
    # First train the models to get their feature importances
    if args.compare_models:
        print("\nTraining models to get feature importances...")
        comparison, rf_importances = train_and_compare_models(output_dir)
        results['model_comparison'] = comparison
        
        # Print results
        print(f"\nRandom Forest Model Comparison Results:")
        print(f"Full model (all features):")
        print(f"  F1 Score: {comparison['random_forest']['full_model']['f1']:.3f}")
        print(f"  AUC: {comparison['random_forest']['full_model']['auc']:.3f}")
        print(f"Selected features model:")
        print(f"  F1 Score: {comparison['random_forest']['selected_model']['f1']:.3f}")
        print(f"  AUC: {comparison['random_forest']['selected_model']['auc']:.3f}")
        print(f"Number of features reduced from {comparison['random_forest']['num_features']['full']} to {comparison['random_forest']['num_features']['selected']}")
        print("\nSelected features:")
        for feature in comparison['random_forest']['selected_features']:
            print(f"  - {feature}")

    trial_importance = agg_trial_feature_importances()
    results['trial_importance'] = trial_importance
    plot_feature_importance(trial_importance,
                          'Random Forest Feature Importance',
                          os.path.join(output_dir, 'rf.png'))

    if args.compare_models:
        # Compute combined feature importance including both models
        combined_importance = compute_combined_feature_importance(
            trial_importance,
            rf_importances,
            comparison['random_forest']['selected_features']
        )
        results['combined_importance'] = combined_importance
        plot_combined_importance(combined_importance, os.path.join(output_dir, 'combined.png'))
        
        # Select important features for combined importance
        selected_combined = select_important_features(combined_importance, args.importance_threshold)
        results['selected_combined_features'] = selected_combined
        plot_cumulative_importance(selected_combined, 
                                 os.path.join(output_dir, 'combined_cumulative.png'))

    # Save results to JSON
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)