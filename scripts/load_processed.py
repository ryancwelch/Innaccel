#!/usr/bin/env python
import os
import numpy as np
import argparse
import sys

# Function to detect if running in a notebook
def is_running_in_notebook():
    try:
        from IPython import get_ipython
        if 'IPKernelApp' in get_ipython().config:
            return True
    except:
        pass
    return False

# Find the project root directory regardless of where the script is running from
def find_project_root():
    # Start with current working directory
    cwd = os.getcwd()
    
    # If we're in a notebook in a subdirectory (like notebooks/), go up one level
    if is_running_in_notebook() and ('notebooks' in cwd or 'notebook' in cwd):
        return os.path.dirname(cwd)
    
    # Check if we already have the data directory where expected
    if os.path.exists(os.path.join(cwd, 'data', 'processed')):
        return cwd
    
    # Try going up one directory if data isn't found
    parent_dir = os.path.dirname(cwd)
    if os.path.exists(os.path.join(parent_dir, 'data', 'processed')):
        return parent_dir
    
    # Default to current directory if we can't find the data
    return cwd

# Set the project root directory
PROJECT_ROOT = find_project_root()

def get_absolute_path(relative_path):
    """Convert a relative path to absolute based on project root"""
    if os.path.isabs(relative_path):
        return relative_path
    return os.path.join(PROJECT_ROOT, relative_path)

def load_processed_data(record_name, processed_dir="data/processed", verbose=True):
    """
    Loads processed EHG signals and processing information for a specified record.
    
    Parameters:
    -----------
    record_name : str
        Name of the record to load
    processed_dir : str
        Directory containing the processed signals (default: data/processed)
        Use data/processed_nar for signals processed without artifact removal
    verbose : bool
        Whether to print information during loading
        
    Returns:
    --------
    processed_signals : np.ndarray
        The processed EHG signals
    processing_info : dict
        Dictionary containing processing information and parameters
    """
    # Convert relative path to absolute path
    processed_dir = get_absolute_path(processed_dir)
    
    # Show current directory and the directory we're trying to access
    if verbose:
        print(f"Project root: {PROJECT_ROOT}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Looking for data in: {processed_dir}")
        print(f"Directory exists: {os.path.exists(processed_dir)}")
        
        # If directory doesn't exist, suggest similar directories
        if not os.path.exists(processed_dir):
            # Check for common parent directories
            parent_dirs = [
                PROJECT_ROOT,
                os.path.dirname(os.getcwd()),
                os.path.join(os.path.dirname(os.getcwd()), "data"),
                os.path.join(PROJECT_ROOT, "data")
            ]
            
            for parent in parent_dirs:
                if os.path.exists(parent):
                    print(f"Parent directory exists: {parent}")
                    possible_dirs = [d for d in os.listdir(parent) 
                                    if os.path.isdir(os.path.join(parent, d))]
                    print(f"Possible subdirectories: {possible_dirs}")
    
    # Construct file paths
    signals_path = os.path.join(processed_dir, f"{record_name}_processed.npy")
    info_path = os.path.join(processed_dir, f"{record_name}_processing_info.npy")
    
    # Check if files exist
    if not os.path.exists(signals_path):
        if verbose:
            print(f"Error: Processed signals file not found at {signals_path}")
            
            # Try to find matching files
            if os.path.exists(processed_dir):
                files = os.listdir(processed_dir)
                matching_files = [f for f in files if record_name in f]
                if matching_files:
                    print(f"Found similar files in {processed_dir}: {matching_files}")
                else:
                    print(f"No files matching '{record_name}' found in {processed_dir}")
                    all_npys = [f for f in files if f.endswith('.npy')]
                    if all_npys:
                        print(f"Available .npy files: {all_npys[:10]} {'...' if len(all_npys) > 10 else ''}")
        return None, None
    
    if not os.path.exists(info_path):
        if verbose:
            print(f"Error: Processing info file not found at {info_path}")
        return None, None
    
    # Load the data
    try:
        processed_signals = np.load(signals_path)
        processing_info = np.load(info_path, allow_pickle=True).item()
        
        if verbose:
            print(f"Loaded processed signals with shape {processed_signals.shape}")
            print(f"Sampling rate: {processing_info.get('processed_fs', 'unknown')} Hz")
            print(f"Processing steps: {processing_info.get('processing_steps', [])}")
        
        return processed_signals, processing_info
    
    except Exception as e:
        if verbose:
            print(f"Error loading processed data: {e}")
        return None, None

def load_all_processed_records(processed_dir="data/processed", verbose=True):
    """
    Loads all processed records from the specified directory.
    
    Parameters:
    -----------
    processed_dir : str
        Directory containing the processed signals
    verbose : bool
        Whether to print information during loading
        
    Returns:
    --------
    processed_data : dict
        Dictionary mapping record names to (signals, info) tuples
    """
    # Convert to absolute path
    processed_dir = get_absolute_path(processed_dir)
    
    # Check if directory exists
    if not os.path.exists(processed_dir):
        if verbose:
            print(f"Error: Processed data directory not found at {processed_dir}")
            
            # Suggest potential alternative directories
            parent_dir = os.path.dirname(processed_dir)
            if os.path.exists(parent_dir):
                subdirs = [d for d in os.listdir(parent_dir) 
                          if os.path.isdir(os.path.join(parent_dir, d))]
                if subdirs:
                    print(f"Available directories in {parent_dir}: {subdirs}")
        return {}
    
    # Find all processed signal files
    signal_files = [f for f in os.listdir(processed_dir) if f.endswith("_processed.npy")]
    record_names = [f.replace("_processed.npy", "") for f in signal_files]
    
    if verbose:
        print(f"Found {len(record_names)} processed records in {processed_dir}")
        if record_names:
            print(f"Available records: {record_names}")
    
    # Load each record
    processed_data = {}
    for record_name in record_names:
        signals, info = load_processed_data(record_name, processed_dir, verbose=verbose)
        if signals is not None and info is not None:
            processed_data[record_name] = (signals, info)
    
    if verbose:
        print(f"Successfully loaded {len(processed_data)} records")
    
    return processed_data

def load_processing_summary(processed_dir="data/processed", verbose=True):
    """
    Load the processing summary file that contains information about all processed records.
    
    Parameters:
    -----------
    processed_dir : str
        Directory containing the processed data
    verbose : bool
        Whether to print information during loading
        
    Returns:
    --------
    summary : dict
        Dictionary containing summary information for all processed records
    """
    # Convert to absolute path
    processed_dir = get_absolute_path(processed_dir)
    
    summary_path = os.path.join(processed_dir, "all_processing_summary.npy")
    
    if not os.path.exists(summary_path):
        if verbose:
            print(f"Warning: Processing summary file not found at {summary_path}")
        return None
    
    try:
        summary = np.load(summary_path, allow_pickle=True).item()
        
        if verbose:
            print(f"Loaded processing summary with {len(summary)} records")
        
        return summary
    
    except Exception as e:
        if verbose:
            print(f"Error loading processing summary: {e}")
        return None

def compare_processing_methods(record_name, verbose=True):
    """
    Compare processed signals with and without artifact removal for a specified record.
    
    Parameters:
    -----------
    record_name : str
        Name of the record to compare
    verbose : bool
        Whether to print information during loading
        
    Returns:
    --------
    standard_data : tuple
        (signals, info) for standard processing
    nar_data : tuple
        (signals, info) for NAR processing
    """
    # Load data from both processing methods
    standard_signals, standard_info = load_processed_data(
        record_name, 
        processed_dir="data/processed",
        verbose=verbose
    )
    
    nar_signals, nar_info = load_processed_data(
        record_name, 
        processed_dir="data/processed_nar",
        verbose=verbose
    )
    
    # Print comparison if verbose
    if verbose and standard_signals is not None and nar_signals is not None:
        print("\nComparison:")
        print(f"Standard processing shape: {standard_signals.shape}")
        print(f"NAR processing shape: {nar_signals.shape}")
        
        # Calculate difference in duration
        if 'processed_fs' in standard_info and 'processed_fs' in nar_info:
            std_duration = standard_signals.shape[0] / standard_info['processed_fs']
            nar_duration = nar_signals.shape[0] / nar_info['processed_fs']
            print(f"Standard duration: {std_duration:.2f}s")
            print(f"NAR duration: {nar_duration:.2f}s")
            print(f"Difference: {nar_duration - std_duration:.2f}s")
    
    return (standard_signals, standard_info), (nar_signals, nar_info)

def list_available_records(verbose=True):
    """List all available processed records in both directories"""
    std_dir = get_absolute_path("data/processed")
    nar_dir = get_absolute_path("data/processed_nar")
    
    print("\nStandard processed records:")
    if os.path.exists(std_dir):
        files = os.listdir(std_dir)
        records = sorted(set([f.split('_processed')[0] for f in files if f.endswith('_processed.npy')]))
        for record in records:
            print(f"  - {record}")
    else:
        print(f"  Directory not found: {std_dir}")
    
    print("\nNAR processed records:")
    if os.path.exists(nar_dir):
        files = os.listdir(nar_dir)
        records = sorted(set([f.split('_processed')[0] for f in files if f.endswith('_processed.npy')]))
        for record in records:
            print(f"  - {record}")
    else:
        print(f"  Directory not found: {nar_dir}")

def get_available_records(processed_dir="data/processed", verbose=True):
    """Returns a list of all available processed records in both directories"""
    std_dir = get_absolute_path(processed_dir)
    nar_dir = get_absolute_path(f"{processed_dir}_nar")
    
    std_records = []
    nar_records = []
    
    if os.path.exists(std_dir):
        files = os.listdir(std_dir)
        records = sorted(set([f.split('_processed')[0] for f in files if f.endswith('_processed.npy')]))
        std_records.extend(records)
    else:
        print(f"  Directory not found: {std_dir}")
    
    print("\nNAR processed records:")
    if os.path.exists(nar_dir):
        files = os.listdir(nar_dir)
        records = sorted(set([f.split('_processed')[0] for f in files if f.endswith('_processed.npy')]))
        nar_records.extend(records)
    else:
        print(f"  Directory not found: {nar_dir}")
    
    return std_records, nar_records

def main():
    parser = argparse.ArgumentParser(
        description='Load processed EHG signals from standard or NAR processing',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--record', type=str, help='Record name to load')
    parser.add_argument('--processed_dir', type=str, default='data/processed',
                      help='Directory containing processed data')
    parser.add_argument('--compare', action='store_true',
                      help='Compare standard and NAR processing for the specified record')
    parser.add_argument('--summary', action='store_true',
                      help='Load and print the processing summary')
    parser.add_argument('--list_all', action='store_true',
                      help='List all processed records in the directory')
    parser.add_argument('--list_available', action='store_true',
                      help='List all available records in both directories')
    
    args = parser.parse_args()
    
    # Process based on arguments
    if args.list_available:
        list_available_records()
    elif args.compare and args.record:
        compare_processing_methods(args.record)
    elif args.record:
        load_processed_data(args.record, args.processed_dir)
    elif args.summary:
        summary = load_processing_summary(args.processed_dir)
        if summary:
            print("\nProcessing Summary:")
            for record, info in summary.items():
                steps = ', '.join(info.get('processing_steps', []))
                shape = info.get('processed_shape', 'unknown')
                print(f"  {record}: {shape}, Steps: {steps}")
    elif args.list_all:
        data = load_all_processed_records(args.processed_dir, verbose=True)
        if data:
            print("\nAvailable Records:")
            for record, (signals, info) in data.items():
                fs = info.get('processed_fs', 'unknown')
                duration = signals.shape[0] / fs if fs != 'unknown' else 'unknown'
                print(f"  {record}: Shape {signals.shape}, Duration: {duration:.2f}s")
    else:
        parser.print_help()

# This allows the module to be imported in notebooks without running main()
if __name__ == "__main__":
    main() 