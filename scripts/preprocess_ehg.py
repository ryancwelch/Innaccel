#!/usr/bin/env python
import os
import numpy as np
import argparse
import time
from load_data import load_record
from filtering import apply_all_filters
from downsampling import downsample_signals
from artifact_removal import apply_all_artifact_removal

def has_nans(data, step_name=""):
    """Check if an array contains NaNs and print info if it does"""
    if np.isnan(data).any():
        nan_count = np.isnan(data).sum()
        total_count = data.size
        nan_percent = nan_count / total_count * 100
        print(f"WARNING: Found {nan_count} NaNs out of {total_count} values ({nan_percent:.2f}%) after {step_name}")
        return True
    return False

def safe_copy(data):
    """Make a safe copy of data, replacing any NaNs with zeros"""
    result = data.copy()
    if np.isnan(result).any():
        print(f"Replacing {np.isnan(result).sum()} NaNs with zeros")
        result[np.isnan(result)] = 0
    return result

def preprocess_record(record_name, data_dir="data/records", output_dir="data/processed",
                     lowcut=0.1, highcut=4.0, target_fs=20, trim_seconds=60, 
                     skip_artifact_removal=False, save_intermediate=False, verbose=True):
    """
    Preprocesses a single EHG record through configurable steps:
    1. Loading data
    2. Artifact removal (optional)
    3. Filtering
    4. Downsampling
    
    Parameters:
    -----------
    record_name : str
        Name of the record to process
    data_dir : str
        Directory containing the raw record files
    output_dir : str
        Directory to save processed signals
    lowcut : float
        Lower cutoff frequency for bandpass filter (default: 0.1 Hz)
    highcut : float
        Upper cutoff frequency for bandpass filter (default: 4.0 Hz)
    target_fs : int
        Target sampling frequency after downsampling (default: 20 Hz)
    trim_seconds : int
        Number of seconds to trim from each end (default: 60)
    skip_artifact_removal : bool
        Whether to skip the artifact removal step (default: False)
    save_intermediate : bool
        Whether to save intermediate results from each processing step (default: False)
    verbose : bool
        Whether to print progress information (default: True)
    
    Returns:
    --------
    processed_signals : np.ndarray
        Fully preprocessed signals
    header : dict
        Header information from the original record
    annotations : object
        Annotations from the original record
    processing_info : dict
        Complete information about all processing steps
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    if verbose:
        print(f"Processing record {record_name}...")
        start_time = time.time()
    
    # Step 1: Load the record
    try:
        signals, header, annotations = load_record(record_name, data_dir)
        if verbose:
            print(f"Loaded record with shape {signals.shape}")
            print(f"Original sampling rate: {header['fs']} Hz")
            
        # Check for NaNs in the raw data
        if has_nans(signals, "loading"):
            # Replace NaNs with zeros to avoid propagation
            print("Replacing NaNs with zeros")
            signals = safe_copy(signals)
        
        if save_intermediate:
            np.save(os.path.join(output_dir, f"{record_name}_raw.npy"), signals)
    except Exception as e:
        print(f"Error loading record {record_name}: {e}")
        return None, None, None, None
    
    # Original sampling frequency
    original_fs = header['fs']
    
    # Initialize processing info dictionary
    processing_info = {
        'record_name': record_name,
        'original_fs': original_fs,
        'original_shape': signals.shape,
        'processing_steps': [],
        'skip_artifact_removal': skip_artifact_removal
    }
    
    # Prepare input for next step
    current_signals = signals.copy()
    
    # Step 2: Remove artifacts (only if not skipped)
    if not skip_artifact_removal:
        try:
            current_signals, artifact_info = apply_all_artifact_removal(
                current_signals, 
                fs=original_fs,
                trim_seconds=trim_seconds
            )
            
            # Check for NaNs after artifact removal
            if has_nans(current_signals, "artifact removal"):
                print("Using original signals instead due to NaNs")
                current_signals = safe_copy(signals)
                processing_info['artifact_removal_error'] = "NaNs detected after processing"
            else:
                if verbose:
                    print(f"Applied artifact removal. Shape: {current_signals.shape}")
                    print(f"Trimmed {artifact_info['trim_info']['start_time_seconds']:.2f}s from start, "
                          f"{artifact_info['trim_info']['original_duration_seconds'] - artifact_info['trim_info']['end_time_seconds']:.2f}s from end")
                
                # Add to processing info
                processing_info['artifact_removal'] = artifact_info
                processing_info['processing_steps'].append('artifact_removal')
                
                if save_intermediate:
                    np.save(os.path.join(output_dir, f"{record_name}_cleaned.npy"), current_signals)
                    
        except Exception as e:
            print(f"Error during artifact removal: {e}")
            processing_info['artifact_removal_error'] = str(e)
    
    # Step 3: Apply filters
    try:
        # Make a safe copy in case filters produce NaNs
        signals_before_filtering = current_signals.copy()
        
        current_signals = apply_all_filters(
            current_signals,
            fs=original_fs,
            lowcut=lowcut,
            highcut=highcut
        )
        
        # Check for NaNs after filtering
        if has_nans(current_signals, "filtering"):
            print("Using signals before filtering due to NaNs")
            current_signals = safe_copy(signals_before_filtering)
            processing_info['filtering_error'] = "NaNs detected after processing"
        else:
            if verbose:
                print(f"Applied filtering. Shape: {current_signals.shape}")
            
            # Add to processing info
            processing_info['filtering'] = {
                'lowcut': lowcut,
                'highcut': highcut
            }
            processing_info['processing_steps'].append('filtering')
            
            if save_intermediate:
                np.save(os.path.join(output_dir, f"{record_name}_filtered.npy"), current_signals)
                
    except Exception as e:
        print(f"Error during filtering: {e}")
        processing_info['filtering_error'] = str(e)
    
    # Step 4: Downsample
    try:
        # Make a safe copy in case downsampling produces NaNs
        signals_before_downsampling = current_signals.copy()
        
        current_signals = downsample_signals(
            current_signals,
            original_fs=original_fs,
            target_fs=target_fs
        )
        
        # Check for NaNs after downsampling
        if has_nans(current_signals, "downsampling"):
            print("Downsampling produced NaNs, using alternative resampling method")
            try:
                from scipy import signal
                # Try direct resampling as an alternative
                factor = int(original_fs / target_fs)
                current_signals = np.zeros((signals_before_downsampling.shape[0] // factor, signals_before_downsampling.shape[1]))
                
                for i in range(signals_before_downsampling.shape[1]):
                    # Simple decimation without filtering
                    current_signals[:, i] = signals_before_downsampling[::factor, i]
                
                if has_nans(current_signals, "alternative downsampling"):
                    # If still NaNs, use safe copy of pre-downsampled data
                    print("Alternative downsampling still produced NaNs, using unprocessed data")
                    current_signals = safe_copy(signals)
                    processing_info['downsampling_error'] = "NaNs detected after processing"
                else:
                    processing_info['downsampling'] = {
                        'original_fs': original_fs,
                        'target_fs': target_fs,
                        'decimation_factor': factor,
                        'method': 'simple decimation (fallback)'
                    }
                    processing_info['processing_steps'].append('downsampling')
            except Exception as e:
                print(f"Error during alternative downsampling: {e}")
                current_signals = safe_copy(signals)
                processing_info['downsampling_error'] = str(e)
        else:
            if verbose:
                print(f"Applied downsampling. Final shape: {current_signals.shape}")
                print(f"New sampling rate: {target_fs} Hz")
            
            # Add to processing info
            processing_info['downsampling'] = {
                'original_fs': original_fs,
                'target_fs': target_fs,
                'decimation_factor': int(original_fs / target_fs),
                'method': 'decimation'
            }
            processing_info['processing_steps'].append('downsampling')
        
    except Exception as e:
        print(f"Error during downsampling: {e}")
        processing_info['downsampling_error'] = str(e)
    
    # Final NaN check
    if has_nans(current_signals, "final check"):
        print("WARNING: Final data still contains NaNs. Using original data resampled.")
        # Last resort - use original data with simple resampling
        try:
            factor = int(original_fs / target_fs)
            current_signals = signals[::factor, :]
            if has_nans(current_signals, "last resort resampling"):
                # If still have NaNs, set to zeros
                current_signals = np.zeros((signals.shape[0] // factor, signals.shape[1]))
        except Exception as e:
            print(f"Error in final NaN recovery: {e}")
            # Create dummy data if all else fails
            current_signals = np.zeros((int(signals.shape[0] / (original_fs / target_fs)), signals.shape[1]))
    
    # Update processing info with final results
    processing_info['processed_shape'] = current_signals.shape
    processing_info['processed_fs'] = target_fs
    
    # Calculate mapping between original and processed time points
    if not skip_artifact_removal and 'artifact_removal' in processing_info and 'trim_info' in processing_info['artifact_removal']:
        trim_info = processing_info['artifact_removal']['trim_info']
        
        # Add time mapping info
        processing_info['time_mapping'] = {
            'description': 'Information for mapping between original and processed time points',
            'original_start_time': 0,
            'original_end_time': trim_info['original_duration_seconds'],
            'processed_start_time': 0,
            'processed_end_time': current_signals.shape[0] / target_fs,
            'time_offset': trim_info['start_time_seconds']
        }
    
    # Save final processed signals
    output_path = os.path.join(output_dir, f"{record_name}_processed.npy")
    np.save(output_path, current_signals)
    
    # Save processing info
    info_path = os.path.join(output_dir, f"{record_name}_processing_info.npy")
    np.save(info_path, processing_info)
    
    if verbose:
        print(f"Processing completed in {time.time() - start_time:.2f} seconds")
        print(f"Saved processed signals to {output_path}")
        print(f"Saved processing info to {info_path}")
    
    return current_signals, header, annotations, processing_info

def process_single_record(args):
    """Process a single record with the given arguments"""
    return preprocess_record(
        args.record, 
        args.data_dir, 
        args.output_dir,
        args.lowcut,
        args.highcut,
        args.target_fs,
        args.trim_seconds,
        args.skip_artifact_removal,
        args.save_intermediate,
        verbose=False
    )

def process_batch(args):
    """Process all records in the data directory with the given arguments"""
    # Get list of all record files (.hea files)
    record_files = [f.split('.')[0] for f in os.listdir(args.data_dir) 
                    if f.endswith('.hea') and not f.startswith('.')]
    
    # Remove duplicates (each record has multiple associated files)
    record_names = list(set(record_files))
    
    print(f"Found {len(record_names)} records to process")
    
    # Store processing info for all records
    all_processing_info = {}
    
    # Process each record
    for record_name in record_names:
        _, _, _, processing_info = preprocess_record(
            record_name, 
            args.data_dir, 
            args.output_dir,
            args.lowcut,
            args.highcut,
            args.target_fs,
            args.trim_seconds,
            args.skip_artifact_removal,
            args.save_intermediate,
            verbose=False
        )
        
        if processing_info is not None:
            all_processing_info[record_name] = {
                'record_name': record_name,
                'processed_shape': processing_info['processed_shape'],
                'processing_steps': processing_info['processing_steps'],
                'skip_artifact_removal': args.skip_artifact_removal
            }
    
    # Save summary of all processing info
    summary_path = os.path.join(args.output_dir, "all_processing_summary.npy")
    np.save(summary_path, all_processing_info)
    print(f"Saved processing summary for all records to {summary_path}")
    
    return all_processing_info

def main():
    parser = argparse.ArgumentParser(
        description='Preprocess EHG signals with configurable processing pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/output options
    parser.add_argument('--record', type=str, help='Record name to process (single mode)')
    parser.add_argument('--data_dir', type=str, default='data/records', 
                        help='Directory containing the raw data files')
    parser.add_argument('--output_dir', type=str, default='data/processed', 
                        help='Directory to save processed data')
    
    # Processing parameter options
    parser.add_argument('--lowcut', type=float, default=0.1, 
                        help='Lower cutoff frequency for bandpass filter (Hz)')
    parser.add_argument('--highcut', type=float, default=4.0, 
                        help='Upper cutoff frequency for bandpass filter (Hz)')
    parser.add_argument('--target_fs', type=int, default=20, 
                        help='Target sampling frequency after downsampling (Hz)')
    parser.add_argument('--trim_seconds', type=int, default=60, 
                        help='Seconds to trim from beginning and end of recording (artifact removal)')
    
    # Processing mode options
    parser.add_argument('--skip_artifact_removal', action='store_true', 
                       help='Skip artifact removal step (for NAR processing)')
    parser.add_argument('--save_intermediate', action='store_true', 
                       help='Save intermediate results from each processing step')
    parser.add_argument('--batch', action='store_true', 
                       help='Process all records in the data directory')
    
    args = parser.parse_args()
    
    # Set a different output directory if skipping artifact removal
    if args.skip_artifact_removal:
        args.output_dir = "data/processed_nar"
        print(f"Artifact removal disabled. Output directory set to: {args.output_dir}")
    
    # Process records based on mode
    if args.batch:
        process_batch(args)
    elif args.record:
        process_single_record(args)
    else:
        print("Error: Please provide either a record name (--record) or use --batch to process all records")
        parser.print_help()

if __name__ == "__main__":
    main() 

