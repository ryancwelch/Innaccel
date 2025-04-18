#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import kurtosis, skew
import wfdb
import pywt
from tqdm import tqdm
from load_data import load_record
from load_processed import load_all_processed_records, load_processed_data, get_available_records
import argparse


def calculate_propagation_features(window_data, fs=20):
    """
    Calculate propagation features between channels.
    Assumes channels represent different spatial locations.
    """
    n_channels = window_data.shape[1]
    prop_features = {}
    
    for i in range(n_channels):
        for j in range(i+1, n_channels):
            # Cross-correlation to find time delay
            corr = signal.correlate(window_data[:, i], window_data[:, j], mode='full')
            lags = np.arange(-(len(window_data)-1), len(window_data))
            lag = lags[np.argmax(corr)] / fs  # Convert to seconds
            
            # Assuming 3.5cm spacing between electrodes (adjust based on actual setup)
            electrode_spacing = 0.035  # meters
            if lag != 0:
                velocity = electrode_spacing / abs(lag)  # m/s
            else:
                velocity = 0
                
            prop_features[f'velocity_ch{i}_ch{j}'] = velocity
            prop_features[f'lag_ch{i}_ch{j}'] = lag
            prop_features[f'max_corr_ch{i}_ch{j}'] = np.max(corr)
    
    return prop_features

def extract_window_features(window_data, fs=20):
    """Extract features from a single window of signal data."""
    features = {}
    window_length = window_data.shape[0]
    
    # Use appropriate window size for spectral analysis
    nperseg = min(window_length, 128)  # Smaller window size, must be <= window_length
    
    # Get propagation features first
    prop_features = calculate_propagation_features(window_data, fs)
    features.update(prop_features)
    
    for ch in range(window_data.shape[1]):
        ch_data = window_data[:, ch]
        
        # Time domain features
        features[f'mean_ch{ch}'] = np.mean(ch_data)
        features[f'std_ch{ch}'] = np.std(ch_data)
        features[f'rms_ch{ch}'] = np.sqrt(np.mean(ch_data**2))
        features[f'kurtosis_ch{ch}'] = kurtosis(ch_data)
        features[f'skewness_ch{ch}'] = skew(ch_data)
        features[f'max_amp_ch{ch}'] = np.max(np.abs(ch_data))
        features[f'peak_to_peak_ch{ch}'] = np.max(ch_data) - np.min(ch_data)
        
        # Modified frequency domain analysis
        f, psd = signal.welch(ch_data, fs=fs, nperseg=nperseg)
        
        # Peak frequency and related features
        peak_idx = np.argmax(psd)
        features[f'peak_freq_ch{ch}'] = f[peak_idx]
        features[f'peak_power_ch{ch}'] = psd[peak_idx]
        
        # Frequency bands energy
        features[f'energy_0.1_0.3Hz_ch{ch}'] = np.sum(psd[(f >= 0.1) & (f <= 0.3)])
        features[f'energy_0.3_1Hz_ch{ch}'] = np.sum(psd[(f >= 0.3) & (f <= 1.0)])
        features[f'energy_1_3Hz_ch{ch}'] = np.sum(psd[(f >= 1.0) & (f <= 3.0)])
        
        # Median frequency (frequency below which 50% of signal power exists)
        cum_power = np.cumsum(psd)
        median_freq_idx = np.where(cum_power >= cum_power[-1]/2)[0][0]
        features[f'median_freq_ch{ch}'] = f[median_freq_idx]
        
        # Mean frequency
        features[f'mean_freq_ch{ch}'] = np.sum(f * psd) / np.sum(psd)
        
        # Spectral edge frequencies (90% and 95% of power)
        edge_90_idx = np.where(cum_power >= cum_power[-1]*0.9)[0][0]
        edge_95_idx = np.where(cum_power >= cum_power[-1]*0.95)[0][0]
        features[f'spectral_edge_90_ch{ch}'] = f[edge_90_idx]
        features[f'spectral_edge_95_ch{ch}'] = f[edge_95_idx]
        
        # Spectral entropy
        psd_norm = psd / np.sum(psd)
        spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
        features[f'spectral_entropy_ch{ch}'] = spectral_entropy
        
        # Wavelet features
        coeffs = pywt.wavedec(ch_data, 'db4', level=4)
        for i, coeff in enumerate(coeffs):
            features[f'wavelet_energy_level{i}_ch{ch}'] = np.sum(coeff**2) / len(coeff)
        
        # Add cross-channel frequency coupling (if not first channel)
        if ch > 0:
            # Calculate coherence with previous channel
            f, coherence = signal.coherence(
                window_data[:, ch-1], 
                ch_data, 
                fs=fs, 
                nperseg=nperseg
            )
            features[f'mean_coherence_ch{ch-1}_ch{ch}'] = np.mean(coherence)
            features[f'max_coherence_ch{ch-1}_ch{ch}'] = np.max(coherence)
            
            # Find frequency of maximum coherence
            max_coh_freq = f[np.argmax(coherence)]
            features[f'max_coherence_freq_ch{ch-1}_ch{ch}'] = max_coh_freq
    
    return features

def create_contraction_labels(annotations, signal_length, fs):
    """Create binary labels for contractions."""
    labels = np.zeros(signal_length)
    
    for idx, symbol in enumerate(annotations.symbol):
        if symbol in ['C', '(c)']:
            start_sample = annotations.sample[idx]
            
            # Estimate contraction duration
            if idx < len(annotations.sample) - 1:
                end_sample = annotations.sample[idx + 1]
            else:
                end_sample = start_sample + int(45 * annotations.fs)  # 45s default duration
            
            # Convert to downsampled timeline
            start_idx = start_sample // (annotations.fs // fs)
            end_idx = min(end_sample // (annotations.fs // fs), signal_length)
            
            # Mark contraction period
            labels[start_idx:end_idx] = 1
    
    return labels

def prepare_sequence_data(record_name, data_dir="data/records", window_size=10, step_size=1):
    """Prepare sequence data for LSTM training."""
    # Load preprocessed data
    signals, processing_info = load_processed_data(record_name, processed_dir="data/processed")
    if signals is None:
        raise ValueError(f"Could not load processed data for record {record_name}")
    
    # Get annotations from original data (since we need them for labels)
    _, header, annotations = load_record(record_name, data_dir)
    if annotations is None:
        print(f"Could not load annotations for record {record_name}")
        return None, None
    # Create labels (using the processed signal's sampling rate)
    fs = processing_info['processed_fs']  # Should be 20 Hz
    labels = create_contraction_labels(annotations, len(signals), fs)
    
    # Calculate window parameters
    samples_per_window = window_size * fs
    step_samples = step_size * fs
    n_windows = (len(signals) - samples_per_window) // step_samples + 1
    
    # Initialize lists for sequence data
    X = []  # Features
    y = []  # Labels
    
    # Extract features for each window
    for i in tqdm(range(n_windows), desc=f"Preparing {record_name}"):
        start_idx = i * step_samples
        end_idx = start_idx + samples_per_window
        
        # Extract window data
        window_data = signals[start_idx:end_idx]
        window_features = extract_window_features(window_data, fs=fs)
        
        # Get label for this window (majority vote)
        window_label = int(np.mean(labels[start_idx:end_idx]) > 0.5)
        
        X.append(list(window_features.values()))
        y.append(window_label)
    
    return np.array(X), np.array(y)

def get_feature_names(n_channels):
    """Get list of all feature names that will be generated."""
    feature_names = []
    
    # Propagation features
    for i in range(n_channels):
        for j in range(i+1, n_channels):
            feature_names.extend([
                f'velocity_ch{i}_ch{j}',
                f'lag_ch{i}_ch{j}',
                f'max_corr_ch{i}_ch{j}'
            ])
    
    # Channel-specific features
    for ch in range(n_channels):
        # Time domain features
        feature_names.extend([
            f'mean_ch{ch}',
            f'std_ch{ch}',
            f'rms_ch{ch}',
            f'kurtosis_ch{ch}',
            f'skewness_ch{ch}',
            f'max_amp_ch{ch}',
            f'peak_to_peak_ch{ch}'
        ])
        
        # Frequency domain features
        feature_names.extend([
            f'peak_freq_ch{ch}',
            f'peak_power_ch{ch}',
            f'energy_0.1_0.3Hz_ch{ch}',
            f'energy_0.3_1Hz_ch{ch}',
            f'energy_1_3Hz_ch{ch}',
            f'median_freq_ch{ch}',
            f'mean_freq_ch{ch}',
            f'spectral_edge_90_ch{ch}',
            f'spectral_edge_95_ch{ch}',
            f'spectral_entropy_ch{ch}'
        ])
        
        # Wavelet features
        for i in range(5):  # 4 levels + approximation
            feature_names.append(f'wavelet_energy_level{i}_ch{ch}')
        
        # Cross-channel features (if not first channel)
        if ch > 0:
            feature_names.extend([
                f'mean_coherence_ch{ch-1}_ch{ch}',
                f'max_coherence_ch{ch-1}_ch{ch}',
                f'max_coherence_freq_ch{ch-1}_ch{ch}'
            ])
    
    return feature_names

def main():
    parser = argparse.ArgumentParser(
        description='Prepare EHG sequence data for LSTM training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Directory arguments
    parser.add_argument('--data_dir', type=str, default='data/records',
                      help='Directory containing the raw data files')
    parser.add_argument('--processed_dir', type=str, default='data/processed',
                      help='Directory containing processed signals')
    parser.add_argument('--output_dir', type=str, default='data/lstm_data',
                      help='Directory to save LSTM-ready data')
    
    # Window parameters
    parser.add_argument('--window_size', type=int, default=45,
                      help='Window size in seconds')
    parser.add_argument('--step_size', type=int, default=5,
                      help='Step size in seconds (window stride)')
    
    # Processing options
    parser.add_argument('--use_nar', action='store_true',
                      help='Use NAR (No Artifact Removal) processed signals instead of standard processed signals')
    parser.add_argument('--max_records', type=int, default=None,
                      help='Maximum number of records to process (default: all)')
    parser.add_argument('--verbose', action='store_true',
                      help='Print detailed processing information')
    
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs(args.output_dir, exist_ok=True)

    # Get list of available records
    std_records, nar_records = get_available_records()
    records_to_use = nar_records if args.use_nar else std_records
    processed_dir = 'data/processed_nar' if args.use_nar else args.processed_dir

    if not records_to_use:
        print(f"No {'NAR' if args.use_nar else 'standard'} processed records found! Please run preprocess_ehg.py first.")
        return
    
    print(f"Found {len(std_records)} standard processed records")
    print(f"Found {len(nar_records)} NAR processed records")
    print(f"Using {'NAR' if args.use_nar else 'standard'} processed records")
    
    # Load first record to get number of channels and features
    signals, _ = load_processed_data(records_to_use[0], processed_dir)
    n_channels = signals.shape[1]
    
    # Get feature names
    feature_names = get_feature_names(n_channels)
    n_features = len(feature_names)
    
    # Save feature names
    output_prefix = 'nar_' if args.use_nar else ''
    with open(os.path.join(args.output_dir, f'{output_prefix}feature_names.txt'), 'w') as f:
        for name in feature_names:
            f.write(f"{name}\n")
    
    # Limit number of records if specified
    if args.max_records is not None:
        records_to_use = records_to_use[:args.max_records]
        print(f"Limited to processing {args.max_records} records")
    
    # Process each record and track sequence lengths
    all_sequences = []
    all_labels = []
    sequence_lengths = []
    total_positive = 0
    total_samples = 0
    
    for record in tqdm(records_to_use, desc="Processing records"):
        try:
            if args.verbose:
                print(f"\nProcessing record: {record}")
            X, y = prepare_sequence_data(
                record, 
                data_dir=args.data_dir,
                window_size=args.window_size,
                step_size=args.step_size
            )
            if X is None or y is None:
                print(f"Skipping record {record} due to missing annotations")
                continue
                
            all_sequences.append(X)
            all_labels.append(y)
            sequence_lengths.append(len(X))
            total_positive += np.sum(y)
            total_samples += len(y)
            
            if args.verbose:
                print(f"Successfully processed {record}")
        except Exception as e:
            print(f"Error processing record {record}: {str(e)}")
            continue
    
    if not all_sequences:
        print("No records were successfully processed!")
        return
    
    # Find maximum sequence length
    max_length = max(sequence_lengths)
    
    # Create padded arrays
    X_padded = np.zeros((len(records_to_use), max_length, n_features))
    y_padded = np.zeros((len(records_to_use), max_length))
    
    # Fill the padded arrays
    for i, (X, y) in enumerate(zip(all_sequences, all_labels)):
        seq_length = len(X)
        X_padded[i, :seq_length, :] = X
        y_padded[i, :seq_length] = y
    
    # Save as numpy arrays
    np.save(os.path.join(args.output_dir, f'{output_prefix}X_sequence.npy'), X_padded)
    np.save(os.path.join(args.output_dir, f'{output_prefix}y_sequence.npy'), y_padded)
    np.save(os.path.join(args.output_dir, f'{output_prefix}sequence_lengths.npy'), np.array(sequence_lengths))

    # Save configuration
    config = {
        'window_size': args.window_size,
        'step_size': args.step_size,
        'processed_dir': processed_dir,
        'use_nar': args.use_nar,
        'num_records': len(records_to_use),
        'data_shape': X_padded.shape,
        'max_sequence_length': max_length,
        'sequence_lengths': sequence_lengths,
        'processed_records': records_to_use
    }
    np.save(os.path.join(args.output_dir, f'{output_prefix}config.npy'), config)

    print(f"\nProcessing Summary:")
    print(f"Successfully processed {len(records_to_use)} records")
    print(f"Window size: {args.window_size}s, Step size: {args.step_size}s")
    print(f"Final sequence data shape (n_records, max_seq_len, n_features): {X_padded.shape}")
    print(f"Maximum sequence length: {max_length}")
    print(f"Sequence lengths range: {min(sequence_lengths)} - {max(sequence_lengths)}")
    print(f"Total samples: {total_samples}")
    print(f"Positive samples (contractions): {total_positive}/{total_samples}")
    print(f"Data saved to: {args.output_dir}")

if __name__ == "__main__":
    main() 