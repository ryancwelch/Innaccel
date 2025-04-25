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
from envelop_filtering import calculate_envelope_features
from load_contractions import load_contraction_annotations_from_csv

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
    
    # Get envelope features for each channel
    for ch in range(window_data.shape[1]):
        ch_data = window_data[:, ch]

    
    for ch in range(window_data.shape[1]):
        ch_data = window_data[:, ch]

        # envelope features
        envelope_features = calculate_envelope_features(ch_data, fs)
        ch_envelope_features = {f'{k}_ch{ch}': v for k, v in envelope_features.items()}
        features.update(ch_envelope_features)
        
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
        
        # Envelope features
        feature_names.extend([
            f'envelope_upper_mean_ch{ch}',
            f'envelope_upper_std_ch{ch}',
            f'envelope_lower_mean_ch{ch}',
            f'envelope_lower_std_ch{ch}',
            f'envelope_range_mean_ch{ch}',
            f'envelope_range_std_ch{ch}',
            f'envelope_symmetry_ch{ch}',
            f'area_coefficient_mean_ch{ch}',
            f'area_coefficient_std_ch{ch}',
            f'cosine_similarity_mean_ch{ch}',
            f'cosine_similarity_std_ch{ch}',
            f'rectangle_index_mean_ch{ch}',
            f'rectangle_index_std_ch{ch}'
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
    # Load annotations from contractions.csv
    annotations_dict = load_contraction_annotations_from_csv("contractions.csv")
    records = list(annotations_dict.keys())
    
    # Select one record
    record_name = records[0]  # Use the first record
    print(f"Testing feature extraction on record: {record_name}")
    
    # Load processed data for the record
    signals, processing_info = load_processed_data(record_name, processed_dir="data/processed")
    if signals is None:
        print(f"Could not load processed data for {record_name}")
        exit(1)
    
    # Get annotations for this record
    contractions = annotations_dict[record_name]
    print(f"Found {len(contractions)} contractions in record")
    
    # Process each channel
    for ch in range(signals.shape[1]):
        print(f"\nProcessing channel {ch+1}")
        signal = signals[:, ch]
        
        # Extract features for the entire signal
        features = extract_window_features(signal.reshape(-1, 1), fs=20)
        
        # Print some key features
        print("\nKey features:")
        print(f"Mean amplitude: {features[f'mean_ch{ch}']:.4f}")
        print(f"Standard deviation: {features[f'std_ch{ch}']:.4f}")
        print(f"Peak frequency: {features[f'peak_freq_ch{ch}']:.4f} Hz")
        print(f"Median frequency: {features[f'median_freq_ch{ch}']:.4f} Hz")
        print(f"Energy in 0.1-0.3Hz band: {features[f'energy_0.1_0.3Hz_ch{ch}']:.4f}")
        print(f"Energy in 0.3-1Hz band: {features[f'energy_0.3_1Hz_ch{ch}']:.4f}")
        print(f"Energy in 1-3Hz band: {features[f'energy_1_3Hz_ch{ch}']:.4f}")
        
        # Print envelope features
        print("\nEnvelope features:")
        print(f"Upper envelope mean: {features[f'envelope_upper_mean_ch{ch}']:.4f}")
        print(f"Upper envelope std: {features[f'envelope_upper_std_ch{ch}']:.4f}")
        print(f"Lower envelope mean: {features[f'envelope_lower_mean_ch{ch}']:.4f}")
        print(f"Lower envelope std: {features[f'envelope_lower_std_ch{ch}']:.4f}")
        print(f"Envelope range mean: {features[f'envelope_range_mean_ch{ch}']:.4f}")
        print(f"Envelope range std: {features[f'envelope_range_std_ch{ch}']:.4f}")
        print(f"Envelope symmetry: {features[f'envelope_symmetry_ch{ch}']:.4f}")
        
        # Print area-based features
        print("\nArea-based features:")
        print(f"Area coefficient mean: {features[f'area_coefficient_mean_ch{ch}']:.4f}")
        print(f"Area coefficient std: {features[f'area_coefficient_std_ch{ch}']:.4f}")
        print(f"Cosine similarity mean: {features[f'cosine_similarity_mean_ch{ch}']:.4f}")
        print(f"Cosine similarity std: {features[f'cosine_similarity_std_ch{ch}']:.4f}")
        print(f"Rectangle index mean: {features[f'rectangle_index_mean_ch{ch}']:.4f}")
        print(f"Rectangle index std: {features[f'rectangle_index_std_ch{ch}']:.4f}")
        break

if __name__ == "__main__":
    main() 