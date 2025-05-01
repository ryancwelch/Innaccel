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

# Set the style and font
sns.set(context="talk")
plt.rcParams['font.family'] = 'Arial'

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

def load_trial_results(data_dir: str = "analysis_results") -> pd.DataFrame:
    """Load trial results from the combined results file."""
    results_path = os.path.join(data_dir, 'combined_trial_results.json')
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Combined trial results file not found at {results_path}")
    
    with open(results_path, 'r') as f:
        trial_data = json.load(f)
    
    # Flatten the nested structure
    flattened_data = []
    for trial_id, trial_info in trial_data.items():
        # Combine parameters and results into a single record
        record = {
            'trial_id': trial_id,
            **trial_info['parameters.json'],
            **trial_info['trial_results.json']
        }
        flattened_data.append(record)
    
    return pd.DataFrame(flattened_data)

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
    
    # Extract F1 scores from the nested random_forest_metrics
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
    # Create figures subdirectory
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Extract F1 scores
    df['f1_score'] = df['random_forest_metrics'].apply(lambda x: x['f1'])
    
    # Set the color palette
    sns.set_palette("husl")
    
    # Plot 1: Distribution of F1 scores with KDE
    plt.figure(figsize=(12, 8))
    sns.histplot(df['f1_score'], kde=True, bins=20)
    plt.title('Distribution of F1 Scores Across Hyperparameter Search Trials', fontsize=16, pad=20)
    plt.xlabel('F1 Score', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'f1_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Hyperparameter importance with boxplots
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
    
    # Calculate Cohen's d for each parameter
    effect_sizes = {}
    for param in hyperparameters:
        # Get unique values of the parameter
        unique_values = df[param].unique()
        if len(unique_values) >= 2:
            # Find best and worst performing values
            best_value = df.groupby(param)['f1_score'].mean().idxmax()
            worst_value = df.groupby(param)['f1_score'].mean().idxmin()
            
            # Calculate Cohen's d
            effect_size = cohen_d(
                df[df[param] == best_value]['f1_score'],
                df[df[param] == worst_value]['f1_score']
            )
            effect_sizes[param] = effect_size
        else:
            effect_sizes[param] = None
    
    plt.figure(figsize=(15, 12))
    plt.suptitle('Hyperparameter Importance Analysis with Effect Sizes', fontsize=16, y=1.02)
    for i, param in enumerate(hyperparameters, 1):
        plt.subplot(3, 3, i)
        sns.boxplot(x=param, y='f1_score', data=df)
        
        # Add effect size to title
        effect_size = effect_sizes[param]
        if effect_size is not None:
            effect_interpretation = interpret_cohens_d(effect_size)
            title = f'{param}\n(d={effect_size:.2f}, {effect_interpretation})'
        else:
            title = param
            
        plt.title(title, fontsize=14)
        plt.xticks(rotation=45)
        plt.xlabel('')
        plt.ylabel('F1 Score', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'hyperparameter_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Best Parameter Combinations
    # Get top 3 most important parameters based on effect size
    top_params = sorted([(p, abs(e)) for p, e in effect_sizes.items() if e is not None], 
                       key=lambda x: x[1], reverse=True)[:3]
    top_param_names = [p[0] for p in top_params]
    
    if len(top_param_names) >= 2:
        # Set the style and font
        sns.set(context="talk")
        plt.rcParams['font.family'] = 'Arial'
        
        # Create a figure with a specific layout to accommodate the colorbar
        fig = plt.figure(figsize=(18, 8))
        gs = fig.add_gridspec(2, 3, height_ratios=[15, 1], hspace=0.3, wspace=0.4)
        
        # Find the global min and max F1 scores for consistent coloring
        f1_min = df['f1_score'].min()
        f1_max = df['f1_score'].max()
        
        for i, (param1, param2) in enumerate([(top_param_names[0], top_param_names[1]),
                                            (top_param_names[0], top_param_names[2]),
                                            (top_param_names[1], top_param_names[2])]):
            # Create the heatmap in the top row
            ax = fig.add_subplot(gs[0, i])
            pivot = df.pivot_table(values='f1_score', index=param1, columns=param2, aggfunc='mean')
            
            # Create the heatmap with improved formatting
            sns.heatmap(pivot, 
                       annot=True, 
                       fmt='.3f', 
                       cmap='YlOrRd',
                       square=True,
                       linewidths=0.5,
                       linecolor='gray',
                       vmin=f1_min,
                       vmax=f1_max,
                       cbar=False,
                       ax=ax)
            
            # Improve the title and labels
            ax.set_title(f'{param1} vs {param2}', fontsize=14, pad=10)
            ax.set_xlabel(param2, fontsize=12, labelpad=8)
            ax.set_ylabel(param1, fontsize=12, labelpad=8)
            
            # Rotate x-axis labels for better readability
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
            
            # Add grid lines
            ax.grid(True, which='minor', linestyle='-', color='gray', alpha=0.2)
        
        # Add a single colorbar below the plots
        cbar_ax = fig.add_subplot(gs[1, :])
        norm = plt.Normalize(f1_min, f1_max)
        sm = plt.cm.ScalarMappable(cmap='YlOrRd', norm=norm)
        cbar = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal', label='')
        cbar.ax.tick_params(labelsize=10)
        
        plt.suptitle('F1 Score by Data Hyperparameter Combinations', fontsize=16, y=0.99)
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        plt.savefig(os.path.join(figures_dir, 'parameter_interactions.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot 4: Correlation heatmap of hyperparameters
    plt.figure(figsize=(12, 10))
    corr_matrix = df[hyperparameters].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Hyperparameter Correlation Matrix', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'hyperparameter_correlation.png'), dpi=300, bbox_inches='tight')
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