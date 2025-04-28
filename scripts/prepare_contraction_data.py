#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import sys
import concurrent.futures
from functools import partial

# Import custom modules
from load_processed import load_processed_data, get_available_records
from load_contractions import load_contraction_annotations_from_csv, create_contraction_labels
from extract_features import extract_window_features

def _process_window(args):
    """Helper function for parallel processing of windows."""
    window_data, fs, first_stage_percentile, second_stage_multiplier, labels, start_idx, end_idx, label_threshold = args
    try:
        window_features = extract_window_features(
            window_data, 
            fs=fs, 
            first_stage_percentile=first_stage_percentile,
            second_stage_multiplier=second_stage_multiplier
        )
        
        window_label = int(np.mean(labels[start_idx:end_idx]) > label_threshold)
        
        window_info = {
            'start_idx': start_idx,
            'end_idx': end_idx,
            'start_time': start_idx / fs,
            'end_time': end_idx / fs,
            'label': window_label,
            'contraction_percentage': np.mean(labels[start_idx:end_idx]) * 100
        }
        
        return list(window_features.values()), window_label, window_info, list(window_features.keys())
    except Exception as e:
        return None, None, None, None

def prepare_contraction_dataset(record_name, annotations_dict, 
                              processed_dir="data/processed", 
                              window_size=45, step_size=5,
                              label_threshold=0.5,
                              first_stage_percentile=70,
                              second_stage_multiplier=1.2,
                              verbose=False,
                              no_adjust_trim=False,
                              n_workers=None):
    """
    Prepare a dataset for contraction detection using manual annotations from CSV.
    
    Parameters:
    -----------
    record_name : str
        Name of the record to process
    annotations_dict : dict
        Dictionary mapping record names to contraction annotations
    processed_dir : str
        Directory containing processed signals
    window_size : int
        Window size in seconds
    step_size : int
        Step size in seconds
    label_threshold : float
        Threshold for majority vote label assignment (default: 0.5)
    verbose : bool
        Whether to print progress information
    no_adjust_trim : bool
        Whether to avoid adjusting times for artifact removal trimming
    n_workers : int or None
        Number of worker processes to use (None = use all available cores)
        
    Returns:
    --------
    X : np.ndarray
        Features array of shape (n_windows, n_features)
    y : np.ndarray
        Labels array of shape (n_windows,)
    window_info : list
        List of window information dictionaries
    feature_names : list
        List of feature names
    """
    # Load processed data
    signals, processing_info = load_processed_data(record_name, processed_dir=processed_dir)
    if signals is None:
        if verbose:
            print(f"Could not load processed data for record {record_name}")
        return None, None, None, None
    
    # Create labels from annotations
    fs = processing_info['processed_fs']
    if no_adjust_trim:
        labels = create_contraction_labels(record_name, len(signals), fs, annotations_dict, None)
    else:
        labels = create_contraction_labels(record_name, len(signals), fs, annotations_dict, processing_info)
    
    if np.sum(labels) == 0:
        if verbose:
            print(f"No contractions found for record {record_name}")
        return None, None, None, None
    
    # Calculate window parameters
    samples_per_window = int(window_size * fs)
    step_samples = int(step_size * fs)
    n_windows = (len(signals) - samples_per_window) // step_samples + 1
    
    if verbose:
        print(f"Processing record {record_name} with {n_windows} windows")
    
    # Prepare arguments for parallel processing
    window_args = []
    for i in range(n_windows):
        start_idx = i * step_samples
        end_idx = start_idx + samples_per_window
        window_data = signals[start_idx:end_idx]
        window_args.append((
            window_data, fs, first_stage_percentile, second_stage_multiplier,
            labels, start_idx, end_idx, label_threshold
        ))
    
    # Process windows sequentially (parallelization moved to higher level)
    X = []
    y = []
    window_info = []
    feature_names = None
    
    for args in window_args:
        features, label, info, keys = _process_window(args)
        if features is not None:
            X.append(features)
            y.append(label)
            window_info.append(info)
            if feature_names is None:
                feature_names = keys
    
    if not X:
        if verbose:
            print(f"No valid windows generated for {record_name}")
        return None, None, None, None
    
    return np.array(X), np.array(y), window_info, feature_names

def create_train_test_split(X, y, records, test_size=0.2, random_state=42):
    """
    Create a train-test split based on record names.
    
    Parameters:
    -----------
    X : list of np.ndarray
        List of feature arrays for each record
    y : list of np.ndarray
        List of label arrays for each record
    records : list
        List of record names
    test_size : float
        Proportion of records to use for testing
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    X_train, X_test, y_train, y_test : np.ndarray
        Train and test splits
    train_records, test_records : list
        Lists of record names in each split
    """
    import numpy as np
    from sklearn.model_selection import train_test_split
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    records = np.array(records)
    
    # Get unique records
    unique_records = np.unique(records)
    
    # Split records into train and test
    train_records, test_records = train_test_split(
        unique_records, test_size=test_size, random_state=random_state
    )
    
    # Create masks for train and test
    train_mask = np.isin(records, train_records)
    test_mask = np.isin(records, test_records)
    
    # Split data
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    
    return X_train, X_test, y_train, y_test, train_records, test_records

def process_record(record, annotations_dict, processed_dir, label_threshold, window_size, 
                  first_stage_percentile, second_stage_multiplier, step_size, no_adjust_trim, n_workers):
    """Process a single record in parallel."""
    try:
        X, y, window_info, feature_names = prepare_contraction_dataset(
            record, 
            annotations_dict, 
            processed_dir=processed_dir,
            label_threshold=label_threshold,
            window_size=window_size, 
            first_stage_percentile=first_stage_percentile,
            second_stage_multiplier=second_stage_multiplier,
            step_size=step_size,
            verbose=False,  # Disable verbose for individual records
            no_adjust_trim=no_adjust_trim,
            n_workers=n_workers
        )
        
        if X is None or y is None:
            return None
            
        # Count original vs visible contractions
        original_contractions = len(annotations_dict.get(record, []))
        
        # Load data again to compute visible contractions
        signals, processing_info = load_processed_data(record, processed_dir=processed_dir)
        fs = processing_info['processed_fs']
        labels = create_contraction_labels(record, len(signals), fs, annotations_dict, 
                                         None if no_adjust_trim else processing_info)
        visible_contractions = np.sum(np.diff(np.concatenate(([0], labels, [0]))) > 0)
            
        record_stats = {
            'windows': len(y),
            'positive_windows': int(np.sum(y)),
            'positive_percentage': (np.sum(y) / len(y)) * 100,
            'original_contractions': original_contractions,
            'visible_contractions': visible_contractions,
            'window_info': window_info
        }
        
        return {
            'record': record,
            'feature_names': feature_names,
            'X': X,
            'y': y,
            'record_stats': record_stats,
            'original_contractions': original_contractions,
            'visible_contractions': visible_contractions
        }
            
    except Exception as e:
        print(f"Error processing record {record}: {e}")
        return None

def prepare_full_dataset(annotations_dict, 
                       processed_dir="data/processed", 
                       output_dir="data/contraction_data",
                       label_threshold=0.5,
                       window_size=45, 
                       step_size=5,
                       first_stage_percentile=70,
                       second_stage_multiplier=1.2,
                       max_records=None,
                       split=True,
                       test_size=0.2,
                       verbose=True,
                       no_adjust_trim=False,
                       n_workers=None):
    """
    Prepare a complete dataset for contraction detection using all available records.
    
    Parameters:
    -----------
    annotations_dict : dict
        Dictionary mapping record names to contraction annotations
    processed_dir : str
        Directory containing processed signals
    output_dir : str
        Directory to save the dataset
    window_size : int
        Window size in seconds
    step_size : int
        Step size in seconds
    max_records : int or None
        Maximum number of records to process
    split : bool
        Whether to create train/test splits
    test_size : float
        Proportion of records to use for testing
    verbose : bool
        Whether to print progress information
    no_adjust_trim : bool
        Whether to avoid adjusting times for artifact removal trimming
    n_workers : int or None
        Number of worker processes to use (None = use all available cores)
        
    Returns:
    --------
    dataset_info : dict
        Information about the created dataset
    """
    # Get list of available records
    std_records, _ = get_available_records(processed_dir=processed_dir)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find records that have both processed data and annotations
    valid_records = []
    for record in std_records:
        if record in annotations_dict:
            valid_records.append(record)
    
    if verbose:
        print(f"Found {len(valid_records)} records with both processed data and annotations")
    
    # Limit number of records if specified
    if max_records:
        valid_records = valid_records[:max_records]
        if verbose:
            print(f"Limited to {max_records} records")
    
    # Process each record in parallel
    all_features = []
    all_labels = []
    all_record_names = []
    record_stats = {}
    
    # Track annotation stats
    total_original_contractions = 0
    total_visible_contractions = 0
    
    # Create partial function with fixed parameters
    process_record_partial = partial(
        process_record,
        annotations_dict=annotations_dict,
        processed_dir=processed_dir,
        label_threshold=label_threshold,
        window_size=window_size,
        first_stage_percentile=first_stage_percentile,
        second_stage_multiplier=second_stage_multiplier,
        step_size=step_size,
        no_adjust_trim=no_adjust_trim,
        n_workers=None  # No nested parallelization
    )
    
    # Process records in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        if verbose:
            results = list(tqdm(
                executor.map(process_record_partial, valid_records),
                total=len(valid_records),
                desc="Processing records"
            ))
        else:
            results = list(executor.map(process_record_partial, valid_records))
    
    # Collect results
    for result in results:
        if result is not None:
            record = result['record']
            X = result['X']
            y = result['y']
            
            # Add to dataset
            for i in range(len(y)):
                all_features.append(X[i])
                all_labels.append(y[i])
                all_record_names.append(record)
            
            # Update stats
            record_stats[record] = result['record_stats']
            total_original_contractions += result['original_contractions']
            total_visible_contractions += result['visible_contractions']
            
            if verbose:
                print(f"Added {len(y)} windows from {record} with {np.sum(y)} positive examples ({np.sum(y)/len(y)*100:.2f}%)")
                print(f"Original contractions: {result['original_contractions']}, Visible in processed data: {result['visible_contractions']}")
    
    # Check if we have any data
    if not all_features:
        print("No valid data generated! Check annotations and processed data.")
        return None
    
    # Convert to numpy arrays
    X = np.array(all_features)
    y = np.array(all_labels)
    records = np.array(all_record_names)
    feature_names = result['feature_names']
    
    # Calculate dataset statistics
    dataset_info = {
        'n_samples': len(y),
        'n_features': X.shape[1],
        'n_positive': int(np.sum(y)),
        'positive_percentage': np.sum(y) / len(y) * 100,
        'window_size': window_size,
        'step_size': step_size,
        'n_records': len(valid_records),
        'record_names': valid_records,
        'feature_names': feature_names,
        'record_stats': record_stats,
        'total_original_contractions': total_original_contractions,
        'total_visible_contractions': total_visible_contractions,
        'adjust_for_trim': not no_adjust_trim
    }
    
    # Save the dataset
    np.save(os.path.join(output_dir, 'X.npy'), X)
    np.save(os.path.join(output_dir, 'y.npy'), y)
    np.save(os.path.join(output_dir, 'records.npy'), records)
    np.save(os.path.join(output_dir, 'feature_names.npy'), feature_names)
    np.save(os.path.join(output_dir, 'dataset_info.npy'), dataset_info)
    
    # Create and save train/test split if requested
    if split:
        X_train, X_test, y_train, y_test, train_records, test_records = create_train_test_split(
            X, y, records, test_size=test_size
        )
        
        split_info = {
            'train_size': len(y_train),
            'test_size': len(y_test),
            'train_positive': int(np.sum(y_train)),
            'test_positive': int(np.sum(y_test)),
            'train_positive_percentage': np.sum(y_train) / len(y_train) * 100,
            'test_positive_percentage': np.sum(y_test) / len(y_test) * 100,
            'train_records': train_records.tolist(),
            'test_records': test_records.tolist()
        }
        
        dataset_info['split_info'] = split_info
        
        # Save splits
        np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
        np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
        np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
        np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
        np.save(os.path.join(output_dir, 'dataset_info.npy'), dataset_info)  # Update with split info
    
    # Print dataset summary
    print("\nDataset Summary:")
    print(f"Total samples: {dataset_info['n_samples']}")
    print(f"Features: {dataset_info['n_features']}")
    print(f"Positive examples: {dataset_info['n_positive']} ({dataset_info['positive_percentage']:.2f}%)")
    print(f"Records: {dataset_info['n_records']}")
    print(f"Original contractions: {total_original_contractions}, Visible in processed data: {total_visible_contractions}")
    print(f"Time adjustment for artifact removal: {'Applied' if not no_adjust_trim else 'Not applied'}")
    print(f"Data saved to: {output_dir}")
    
    if split:
        print("\nSplit Summary:")
        print(f"Train samples: {split_info['train_size']} ({split_info['train_positive_percentage']:.2f}% positive)")
        print(f"Test samples: {split_info['test_size']} ({split_info['test_positive_percentage']:.2f}% positive)")
    
    return dataset_info

def visualize_class_distribution(dataset_info, output_dir=None):
    """Create visualizations of the class distribution in the dataset."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Class distribution by record
    records = []
    positive_percentages = []
    
    for record, stats in dataset_info['record_stats'].items():
        records.append(record)
        positive_percentages.append(stats['positive_percentage'])
    
    # Sort by percentage
    sorted_indices = np.argsort(positive_percentages)
    sorted_records = [records[i] for i in sorted_indices]
    sorted_percentages = [positive_percentages[i] for i in sorted_indices]
    
    # Create figure
    plt.figure(figsize=(12, 8))
    plt.bar(range(len(sorted_records)), sorted_percentages)
    plt.xticks(range(len(sorted_records)), sorted_records, rotation=90)
    plt.xlabel('Record')
    plt.ylabel('Percentage of Positive Windows')
    plt.title('Contraction Percentage by Record')
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'class_distribution.png'), dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    # Overall class distribution
    plt.figure(figsize=(8, 6))
    plt.pie([dataset_info['n_positive'], dataset_info['n_samples'] - dataset_info['n_positive']], 
            labels=['Contraction', 'No Contraction'], 
            autopct='%1.1f%%')
    plt.title('Overall Class Distribution')
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'overall_distribution.png'), dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    # Create contractions visualization
    if 'total_original_contractions' in dataset_info and 'total_visible_contractions' in dataset_info:
        plt.figure(figsize=(10, 6))
        labels = ['Original Annotations', 'Visible in Processed Data']
        values = [dataset_info['total_original_contractions'], dataset_info['total_visible_contractions']]
        plt.bar(labels, values)
        for i, v in enumerate(values):
            plt.text(i, v + 1, str(v), ha='center')
        plt.ylabel('Number of Contractions')
        plt.title('Effect of Processing on Contraction Visibility')
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'contraction_visibility.png'), dpi=300, bbox_inches='tight')
        else:
            plt.show()

def main():
    parser = argparse.ArgumentParser(
        description='Prepare contraction detection dataset from CSV annotations',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--csv_path', type=str, default='contractions.csv',
                      help='Path to the CSV file with contraction annotations')
    parser.add_argument('--processed_dir', type=str, default='data/processed',
                      help='Directory containing processed signals')
    parser.add_argument('--output_dir', type=str, default='data/contraction_data',
                      help='Directory to save the dataset')
    parser.add_argument('--window_size', type=int, default=45,
                      help='Window size in seconds')
    parser.add_argument('--step_size', type=int, default=5,
                      help='Step size in seconds')
    parser.add_argument('--label_threshold', type=float, default=0.5,
                      help='Label threshold for majority vote')
    parser.add_argument('--first_stage_percentile', type=int, default=70,
                      help='First stage percentile for envelope detection')
    parser.add_argument('--second_stage_multiplier', type=float, default=1.2,
                      help='Second stage multiplier for envelope detection')
    parser.add_argument('--max_records', type=int, default=None,
                      help='Maximum number of records to process')
    parser.add_argument('--no_split', action='store_true',
                      help='Do not create train/test splits')
    parser.add_argument('--test_size', type=float, default=0.2,
                      help='Proportion of records to use for testing')
    parser.add_argument('--visualize', action='store_true',
                      help='Create visualizations of the dataset')
    parser.add_argument('--no_adjust_trim', action='store_true',
                      help='Do not adjust contraction times for trimmed seconds')
    parser.add_argument('--n_workers', type=int, default=4,
                      help='Number of worker processes to use')
    
    args = parser.parse_args()
    
    # Load annotations from CSV
    annotations_dict = load_contraction_annotations_from_csv(args.csv_path)
    print(f"Loaded annotations for {len(annotations_dict)} records from CSV")
    
    # Prepare the full dataset
    dataset_info = prepare_full_dataset(
        annotations_dict,
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
        label_threshold=args.label_threshold,
        window_size=args.window_size,
        step_size=args.step_size,
        max_records=args.max_records,
        first_stage_percentile=args.first_stage_percentile,
        second_stage_multiplier=args.second_stage_multiplier,
        split=not args.no_split,
        test_size=args.test_size,
        verbose=True,
        no_adjust_trim=args.no_adjust_trim,
        n_workers=args.n_workers
    )
    
    # Create visualizations if requested
    if args.visualize and dataset_info:
        visualize_class_distribution(dataset_info, args.output_dir)

if __name__ == "__main__":
    main() 