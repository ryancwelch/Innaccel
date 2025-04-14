#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import kurtosis, skew
import wfdb
import pywt
from tqdm import tqdm

def check_record_files(record_name, data_dir="data/records"):
    """Check if all necessary files exist for a record."""
    base_path = os.path.join(data_dir, record_name)
    required_files = ['.dat', '.hea', '.atr']
    return all(os.path.exists(base_path + ext) for ext in required_files)

def get_valid_records(data_dir="data/records"):
    """Get list of records with all required files."""
    all_files = os.listdir(data_dir)
    record_names = set(f.split('.')[0] for f in all_files)
    return [r for r in record_names if check_record_files(r, data_dir)]

def load_and_preprocess(record_name, data_dir="data/records"):
    """Load and preprocess a single record."""
    record_path = os.path.join(data_dir, record_name)
    
    # Load signal and annotations
    signals, header = wfdb.rdsamp(record_path)
    annotations = wfdb.rdann(record_path, 'atr')
    
    # Preprocess signals
    fs_original = header['fs']
    
    # Normalize and filter
    normalized = signals - np.mean(signals, axis=0)
    
    # Bandpass filter (0.1-4.0 Hz)
    nyquist = fs_original / 2
    b, a = signal.butter(3, [0.1/nyquist, 4.0/nyquist], btype='band')
    filtered = signal.filtfilt(b, a, normalized, axis=0)
    
    # Notch filter (50 Hz)
    b_notch, a_notch = signal.iirnotch(50.0, 30, fs_original)
    filtered = signal.filtfilt(b_notch, a_notch, filtered, axis=0)
    
    # Downsample to 20 Hz
    target_fs = 20
    downsample_factor = fs_original // target_fs
    downsampled = signal.decimate(filtered, downsample_factor, axis=0)
    
    return downsampled, target_fs, annotations

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
    # Load and preprocess
    signals, fs, annotations = load_and_preprocess(record_name, data_dir)
    
    # Create labels
    labels = create_contraction_labels(annotations, len(signals), fs)
    
    # Calculate window parameters
    samples_per_window = window_size * fs
    step_samples = step_size * fs
    n_windows = (len(signals) - samples_per_window) // step_samples + 1
    
    # Initialize lists for sequence data
    X = []  # Features
    y = []  # Labels
    
    # Extract features for each window
    for i in tqdm(range(n_windows), desc=f"Processing {record_name}"):
        start_idx = i * step_samples
        end_idx = start_idx + samples_per_window
        
        # Extract window data
        window_data = signals[start_idx:end_idx]
        window_features = extract_window_features(window_data)
        
        # Get label for this window (majority vote)
        window_label = int(np.mean(labels[start_idx:end_idx]) > 0.5)
        
        X.append(list(window_features.values()))
        y.append(window_label)
    
    return np.array(X), np.array(y)

def main():
    data_dir = "data/records"
    output_dir = "data/lstm_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get valid records
    valid_records = get_valid_records(data_dir)
    print(f"Found {len(valid_records)} valid records")
    
    if not valid_records:
        print("No valid records found!")
        return
        
    # Initialize lists for all data
    all_X = []
    all_y = []
    
    # Process each record
    for record in valid_records[:5]:
        X, y = prepare_sequence_data(record, data_dir)
        all_X.append(X)
        all_y.append(y)
    
    # Combine all data
    X_combined = np.vstack(all_X)
    y_combined = np.hstack(all_y)
    
    # Save as numpy arrays
    np.save(os.path.join(output_dir, 'X_sequence.npy'), X_combined)
    np.save(os.path.join(output_dir, 'y_sequence.npy'), y_combined)
    
    # Get feature names using the first record's data
    first_record = valid_records[0]
    signals, fs, _ = load_and_preprocess(first_record, data_dir)
    dummy_window = signals[:fs*10, :]  # 10-second window
    feature_names = list(extract_window_features(dummy_window).keys())
    
    with open(os.path.join(output_dir, 'feature_names.txt'), 'w') as f:
        f.write('\n'.join(feature_names))
    
    print(f"Saved sequence data with shape: {X_combined.shape}")
    print(f"Positive samples (contractions): {np.sum(y_combined)}/{len(y_combined)}")
    print(f"Number of features: {len(feature_names)}")

if __name__ == "__main__":
    main() 