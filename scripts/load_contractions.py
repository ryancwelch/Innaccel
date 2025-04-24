#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from load_processed import load_processed_data, get_available_records
from load_data import load_record

def load_contraction_annotations_from_csv(csv_path="contractions.csv"):
    """
    Load contraction annotations from the CSV file.
    
    Parameters:
    -----------
    csv_path : str
        Path to the CSV file containing contraction annotations
        
    Returns:
    --------
    annotations_by_record : dict
        Dictionary mapping record names to lists of contraction annotations
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        # Create a dictionary to store annotations by record name
        annotations_by_record = {}
        
        # Process each row
        for _, row in df.iterrows():
            # Extract record name from filename (remove .jpg extension)
            record_name = row['Filename'].replace('.jpg', '')
            
            # Check if it's a valid record name format
            if not (record_name.startswith('ice') and ('_p_' in record_name or '_l_' in record_name)):
                print(f"Warning: Unusual record name format: {record_name}")
            
            # Create annotation entry with start and end times
            contraction = {
                'id': int(row['Contraction']),
                'start_time': float(row['Start of contraction (s)']),
                'end_time': float(row['End of contraction (s)']),
                'type': 'C'  # Assuming all are definite contractions
            }
            
            # Add to dictionary
            if record_name not in annotations_by_record:
                annotations_by_record[record_name] = []
            
            annotations_by_record[record_name].append(contraction)
        
        # Sort contractions for each record by start time
        for record_name in annotations_by_record:
            annotations_by_record[record_name].sort(key=lambda x: x['start_time'])
        
        return annotations_by_record
    
    except Exception as e:
        print(f"Error loading contraction annotations: {e}")
        return {}

def create_contraction_labels(record_name, signal_length, fs, annotations_dict, processing_info=None):
    """
    Create binary labels for contractions using the CSV annotations.
    
    Parameters:
    -----------
    record_name : str
        Name of the record
    signal_length : int
        Length of the signal in samples
    fs : float
        Sampling frequency in Hz
    annotations_dict : dict
        Dictionary mapping record names to contraction annotations
    processing_info : dict or None
        Processing information containing trim_seconds if available
        
    Returns:
    --------
    labels : np.ndarray
        Binary array with 1s during contractions and 0s elsewhere
    """
    labels = np.zeros(signal_length)
    
    # Get annotations for this record
    if record_name not in annotations_dict:
        print(f"No annotations found for {record_name} in CSV")
        return labels
    
    contractions = annotations_dict[record_name]
    
    # Get trimming information from processing_info
    trim_seconds = 0
    if processing_info and 'trim_seconds' in processing_info:
        trim_seconds = processing_info['trim_seconds']
    elif processing_info and 'artifact_removal' in processing_info.get('processing_steps', []):
        # If trim_seconds not explicitly provided but artifact removal was performed,
        # use the default value (60 seconds)
        trim_seconds = 60
        
    if trim_seconds > 0:
        print(f"  Adjusting contraction times for {record_name} (trimmed {trim_seconds}s from start)")
    
    # Mark each contraction in the labels array
    for contraction in contractions:
        # Adjust for artifact removal trimming
        start_time = contraction['start_time'] - trim_seconds
        end_time = contraction['end_time'] - trim_seconds
        
        # Convert time to samples
        start_idx = int(start_time * fs)
        end_idx = int(end_time * fs)
        
        # Skip contractions that are entirely before the start of the processed signal
        if end_idx <= 0:
            print(f"  Warning: Contraction {contraction['id']} occurs before processed signal start and will be skipped")
            continue
            
        # Skip contractions that are entirely after the end of the processed signal
        if start_idx >= signal_length:
            print(f"  Warning: Contraction {contraction['id']} occurs after processed signal end and will be skipped")
            continue
        
        # Ensure indices are within bounds
        start_idx = max(0, start_idx)
        end_idx = min(signal_length, end_idx)
        
        # Mark contraction period
        if end_idx > start_idx:
            labels[start_idx:end_idx] = 1
        else:
            print(f"  Warning: Invalid contraction bounds for {record_name}: {start_idx}-{end_idx}")
    
    return labels

def plot_signal_with_contractions(record_name, signals, labels, fs=20, channel_idx=0, save_path=None, processing_info=None):
    """
    Plot the signal with contraction annotations.
    
    Parameters:
    -----------
    record_name : str
        Name of the record
    signals : np.ndarray
        Array of shape (N, channels) containing the EHG signals
    labels : np.ndarray
        Binary array with 1s during contractions and 0s elsewhere
    fs : float
        Sampling frequency in Hz
    channel_idx : int
        Index of the channel to plot
    save_path : str or None
        Path to save the figure, or None to display
    processing_info : dict or None
        Processing information containing additional details
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(15, 6))
    
    # Create time axis
    time = np.arange(len(signals)) / fs
    
    # Plot signal
    ax.plot(time, signals[:, channel_idx], 'b-', alpha=0.7)
    
    # Find contraction regions (transitions in labels)
    changes = np.diff(np.concatenate(([0], labels, [0])))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    
    # Ensure equal number of starts and ends
    n_events = min(len(starts), len(ends))
    
    # Plot each contraction
    for i in range(n_events):
        start_time = starts[i] / fs
        end_time = ends[i] / fs
        
        # Highlight the contraction period
        ax.axvspan(start_time, end_time, color='red', alpha=0.2)
        ax.axvline(start_time, color='red', linestyle='--', linewidth=1)
        ax.axvline(end_time, color='red', linestyle='--', linewidth=1)
    
    # Add labels and title
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    title = f'{record_name} - Channel {channel_idx+1} with Contractions'
    
    # Add processing info to title if available
    if processing_info and 'trim_seconds' in processing_info:
        title += f" (Trimmed {processing_info['trim_seconds']}s)"
    elif processing_info and 'artifact_removal' in processing_info.get('processing_steps', []):
        title += f" (Trimmed 60s)"
    
    ax.set_title(title)
    ax.grid(True)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.2, label=f'Contractions ({n_events})')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Save or show the figure
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.tight_layout()
    
    return fig

def count_contractions_per_record(annotations_dict):
    """Count the number of contractions per record in the annotations dictionary."""
    counts = {}
    for record_name, contractions in annotations_dict.items():
        counts[record_name] = len(contractions)
    
    return counts

def validate_annotations(annotations_dict, processed_dir="data/processed"):
    """
    Validate that all records with annotations have corresponding processed data.
    
    Parameters:
    -----------
    annotations_dict : dict
        Dictionary mapping record names to contraction annotations
    processed_dir : str
        Directory containing processed signals
        
    Returns:
    --------
    valid_records : list
        List of record names that have both annotations and processed data
    """
    std_records, _ = get_available_records()
    
    annotated_records = set(annotations_dict.keys())
    processed_records = set(std_records)
    
    valid_records = list(annotated_records.intersection(processed_records))
    missing_processed = annotated_records - processed_records
    missing_annotations = processed_records - annotated_records
    
    print(f"Records with both annotations and processed data: {len(valid_records)}")
    print(f"Records with annotations but no processed data: {len(missing_processed)}")
    print(f"Records with processed data but no annotations: {len(missing_annotations)}")
    
    if missing_processed:
        print("\nMissing processed data for these annotated records:")
        for record in sorted(missing_processed):
            print(f"  - {record}")
    
    return valid_records

def main():
    parser = argparse.ArgumentParser(
        description='Load and visualize contraction annotations from CSV',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--csv_path', type=str, default='contractions.csv',
                      help='Path to the CSV file containing contraction annotations')
    parser.add_argument('--record', type=str, 
                      help='Specific record to visualize (default: visualize all)')
    parser.add_argument('--channel', type=int, default=0,
                      help='Channel index to visualize (0-indexed)')
    parser.add_argument('--save_dir', type=str, default='results/contractions',
                      help='Directory to save visualization figures')
    parser.add_argument('--processed_dir', type=str, default='data/processed',
                      help='Directory containing processed signals')
    parser.add_argument('--validate', action='store_true',
                      help='Validate annotations against available processed data')
    parser.add_argument('--summary', action='store_true',
                      help='Print summary of contraction annotations')
    parser.add_argument('--no_adjust_trim', action='store_true',
                      help='Do not adjust contraction times for trimmed seconds')
    
    args = parser.parse_args()
    
    # Load annotations from CSV
    annotations_dict = load_contraction_annotations_from_csv(args.csv_path)
    print(f"Loaded annotations for {len(annotations_dict)} records from CSV")
    
    # Print summary if requested
    if args.summary:
        counts = count_contractions_per_record(annotations_dict)
        total_contractions = sum(counts.values())
        
        print(f"\nTotal contractions: {total_contractions}")
        print("\nContractions per record:")
        for record_name, count in sorted(counts.items(), key=lambda x: x[0]):
            print(f"  {record_name}: {count}")
    
    # Validate annotations against processed data if requested
    valid_records = list(annotations_dict.keys())
    if args.validate:
        valid_records = validate_annotations(annotations_dict, args.processed_dir)
    
    # Create save directory if needed
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
    
    # Filter to specific record if provided
    if args.record:
        if args.record in annotations_dict:
            valid_records = [args.record]
        else:
            print(f"Error: Record {args.record} not found in annotations")
            return
    
    # Visualize contractions for selected records
    for record_name in valid_records:
        try:
            # Load processed data
            signals, processing_info = load_processed_data(record_name, processed_dir=args.processed_dir)
            if signals is None:
                print(f"Error: Could not load processed data for {record_name}")
                continue
            
            # Create labels, adjusting for trim_seconds if needed
            fs = processing_info['processed_fs']
            
            if args.no_adjust_trim:
                # Don't adjust for trim_seconds
                labels = create_contraction_labels(record_name, len(signals), fs, annotations_dict, None)
            else:
                # Adjust for trim_seconds
                labels = create_contraction_labels(record_name, len(signals), fs, annotations_dict, processing_info)
            
            # Count contractions
            contraction_count = len(annotations_dict.get(record_name, []))
            visible_contractions = np.sum(np.diff(np.concatenate(([0], labels, [0]))) > 0)
            contraction_seconds = np.sum(labels) / fs
            total_seconds = len(labels) / fs
            contraction_percentage = (contraction_seconds / total_seconds) * 100
            
            print(f"\nProcessing {record_name}:")
            print(f"  Total annotated contractions: {contraction_count}")
            print(f"  Visible in processed data: {visible_contractions}")
            print(f"  Contraction time: {contraction_seconds:.1f}s of {total_seconds:.1f}s ({contraction_percentage:.1f}%)")
            
            # Plot signal with contractions
            if args.save_dir:
                save_path = os.path.join(args.save_dir, f"{record_name}_ch{args.channel+1}_contractions.png")
            else:
                save_path = None
                
            plot_signal_with_contractions(
                record_name, 
                signals, 
                labels, 
                fs=fs, 
                channel_idx=args.channel,
                save_path=save_path,
                processing_info=None if args.no_adjust_trim else processing_info
            )
            
            if save_path:
                print(f"  Saved visualization to {save_path}")
            else:
                plt.show()
                
        except Exception as e:
            print(f"Error processing record {record_name}: {e}")
            continue

if __name__ == "__main__":
    main() 