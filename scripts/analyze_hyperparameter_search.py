#!/usr/bin/env python
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from typing import Dict, List, Tuple

def convert_to_python_types(obj):
    """Convert NumPy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_python_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python_types(item) for item in obj]
    return obj

def cohen_d(x, y):
    """Calculate Cohen's d effect size between two groups."""
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)

def load_trial_results(data_dir: str = "data") -> pd.DataFrame:
    """Load all trial results into a pandas DataFrame."""
    results = []
    
    # Find all trial directories
    trial_dirs = [d for d in os.listdir(data_dir) if d.startswith('trial_')]
    
    for trial_dir in trial_dirs:
        results_path = os.path.join(data_dir, trial_dir, 'trial_results.json')
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                trial_data = json.load(f)
                results.append(trial_data)
    
    return pd.DataFrame(results)

def analyze_hyperparameters(df: pd.DataFrame) -> Dict:
    """Analyze hyperparameter performance and confidence."""
    # Define hyperparameters to analyze
    hyperparameters = [
        'first_stage_percentile',
        'second_stage_multiplier',
        'window_size',
        'step_size',
        'label_threshold',
        'lowcut',
        'highcut',
        'target_fs',
        'skip_artifact_removal'
    ]
    
    # Extract F1 scores
    df['f1_score'] = df['random_forest_metrics'].apply(lambda x: x['f1'])
    
    # Find best overall trial
    best_trial = df.loc[df['f1_score'].idxmax()]
    
    # Calculate confidence metrics
    confidence_metrics = {}
    for param in hyperparameters:
        # Group by parameter value and calculate mean F1 score
        param_performance = df.groupby(param)['f1_score'].agg(['mean', 'std', 'count'])
        
        # Calculate confidence interval
        param_performance['ci_lower'] = param_performance['mean'] - 1.96 * param_performance['std'] / np.sqrt(param_performance['count'])
        param_performance['ci_upper'] = param_performance['mean'] + 1.96 * param_performance['std'] / np.sqrt(param_performance['count'])
        
        # Calculate effect size (Cohen's d) between best and worst performing values
        best_value = param_performance['mean'].idxmax()
        worst_value = param_performance['mean'].idxmin()
        
        if len(df[df[param] == best_value]) > 1 and len(df[df[param] == worst_value]) > 1:
            effect_size = cohen_d(
                df[df[param] == best_value]['f1_score'],
                df[df[param] == worst_value]['f1_score']
            )
        else:
            effect_size = None
        
        confidence_metrics[param] = {
            'best_value': best_value,
            'mean_f1': param_performance.loc[best_value, 'mean'],
            'std_f1': param_performance.loc[best_value, 'std'],
            'ci_lower': param_performance.loc[best_value, 'ci_lower'],
            'ci_upper': param_performance.loc[best_value, 'ci_upper'],
            'effect_size': effect_size,
            'n_trials': param_performance.loc[best_value, 'count']
        }
    
    return {
        'best_trial': best_trial.to_dict(),
        'confidence_metrics': confidence_metrics,
        'overall_stats': {
            'mean_f1': df['f1_score'].mean(),
            'std_f1': df['f1_score'].std(),
            'n_trials': len(df)
        }
    }

def plot_hyperparameter_analysis(df: pd.DataFrame, output_dir: str = "analysis_results"):
    """Create visualizations of hyperparameter performance."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract F1 scores
    df['f1_score'] = df['random_forest_metrics'].apply(lambda x: x['f1'])
    
    # Plot 1: Distribution of F1 scores
    plt.figure(figsize=(10, 6))
    sns.histplot(df['f1_score'], kde=True)
    plt.title('Distribution of F1 Scores Across Trials')
    plt.xlabel('F1 Score')
    plt.ylabel('Count')
    plt.savefig(os.path.join(output_dir, 'f1_distribution.png'))
    plt.close()
    
    # Plot 2: Hyperparameter importance
    hyperparameters = [
        'first_stage_percentile',
        'second_stage_multiplier',
        'window_size',
        'step_size',
        'label_threshold',
        'lowcut',
        'highcut',
        'target_fs',
        'skip_artifact_removal'
    ]
    
    plt.figure(figsize=(12, 8))
    for param in hyperparameters:
        plt.subplot(3, 3, hyperparameters.index(param) + 1)
        sns.boxplot(x=param, y='f1_score', data=df)
        plt.title(f'{param} vs F1 Score')
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hyperparameter_importance.png'))
    plt.close()

def interpret_cohens_d(d):
    """Interpret Cohen's d effect size."""
    if d is None:
        return "Not enough data to calculate"
    abs_d = abs(d)
    if abs_d < 0.2:
        return "Negligible effect"
    elif abs_d < 0.5:
        return "Small effect"
    elif abs_d < 0.8:
        return "Medium effect"
    else:
        return "Large effect"

def get_best_hyperparameters(analysis):
    """Extract and format the best hyperparameters with confidence information."""
    best_params = {}
    for param, metrics in analysis['confidence_metrics'].items():
        effect_size = metrics['effect_size']
        interpretation = interpret_cohens_d(effect_size)
        
        best_params[param] = {
            'value': metrics['best_value'],
            'effect_size': effect_size,
            'interpretation': interpretation,
            'confidence': {
                'mean_f1': metrics['mean_f1'],
                'ci_lower': metrics['ci_lower'],
                'ci_upper': metrics['ci_upper']
            }
        }
    
    return best_params

def main():
    # Load and analyze results
    df = load_trial_results()
    analysis = analyze_hyperparameters(df)
    
    # Create output directory
    output_dir = "analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert NumPy types to Python native types before JSON serialization
    analysis = convert_to_python_types(analysis)
    
    # Save analysis results
    with open(os.path.join(output_dir, 'analysis_results.json'), 'w') as f:
        json.dump(analysis, f, indent=4)
    
    # Create visualizations
    plot_hyperparameter_analysis(df, output_dir)
    
    # Print summary
    print("\nHyperparameter Search Analysis Summary:")
    print(f"Total number of trials analyzed: {analysis['overall_stats']['n_trials']}")
    print(f"Average F1 score across all trials: {analysis['overall_stats']['mean_f1']:.4f} ± {analysis['overall_stats']['std_f1']:.4f}")
    
    print("\nCohen's d Effect Size Interpretation:")
    print("d < 0.2: Negligible effect")
    print("0.2 ≤ d < 0.5: Small effect")
    print("0.5 ≤ d < 0.8: Medium effect")
    print("d ≥ 0.8: Large effect")
    
    print("\nBest Trial Parameters:")
    for param, value in analysis['best_trial'].items():
        if param in analysis['confidence_metrics']:
            print(f"{param}: {value}")
    
    print("\nDetailed Parameter Analysis:")
    best_params = get_best_hyperparameters(analysis)
    for param, info in best_params.items():
        print(f"\n{param}:")
        print(f"  Best value: {info['value']}")
        print(f"  Effect size (Cohen's d): {info['effect_size']:.2f} ({info['interpretation']})")
        print(f"  Mean F1: {info['confidence']['mean_f1']:.4f}")
        print(f"  95% CI: [{info['confidence']['ci_lower']:.4f}, {info['confidence']['ci_upper']:.4f}]")
    
    print("\nRecommended Hyperparameters (based on effect size and confidence):")
    # Sort parameters by effect size (absolute value) in descending order
    sorted_params = sorted(best_params.items(), 
                         key=lambda x: abs(x[1]['effect_size']) if x[1]['effect_size'] is not None else 0,
                         reverse=True)
    
    for param, info in sorted_params:
        if info['effect_size'] is not None and abs(info['effect_size']) >= 0.2:  # Only show parameters with at least small effect
            print(f"{param}: {info['value']} (Effect: {info['interpretation']})")

if __name__ == "__main__":
    main() 