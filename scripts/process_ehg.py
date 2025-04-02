#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.stats import kurtosis, skew
import wfdb
import pywt
import time
import argparse

def load_record(record_name, data_dir="../data"):
    """
    Returns the signals, header, and annotations for a given record.
    Tries multiple paths and methods to load the data.
    """
    possible_paths = [
        os.path.join(data_dir, record_name),
        os.path.join(data_dir, "records", record_name)
    ]
    
    for record_path in possible_paths:
        try:
            print(f"Trying to load record from {record_path}")
            signals, header = wfdb.rdsamp(record_path)
            
            annotations = None
            try:
                annotations = wfdb.rdann(record_path, 'atr')
                print(f"Successfully loaded annotations for {record_path}")
                
                if annotations is not None:
                    symbol_counts = {}
                    for symbol in annotations.symbol:
                        if symbol in symbol_counts:
                            symbol_counts[symbol] += 1
                        else:
                            symbol_counts[symbol] = 1
                    print(f"Annotation summary: {symbol_counts}")
            except Exception as e:
                print(f"No annotations found for {record_path}: {e}")
            
            print(f"Successfully loaded record from {record_path}")
            return signals, header, annotations
        except Exception as e:
            print(f"Error loading from {record_path}: {e}")
    
    print(f"Could not load record {record_name} from any path. Please check the file exists.")
    return None, None, None

def preprocess_signals(signals, fs=200, lowcut=0.1, highcut=4.0, notch_freq=50.0, q=30):
    """
    Preprocess EHG signals with bandpass and notch filtering.
    """
    # First normalize the signal to remove DC offset
    normalized_signals = np.zeros_like(signals)
    for i in range(signals.shape[1]):
        normalized_signals[:, i] = signals[:, i] - np.mean(signals[:, i])
    
    # Apply filtering
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(3, [low, high], btype='band') #bandpass filter
    b_notch, a_notch = signal.iirnotch(notch_freq, q, fs) #notch filter
    
    filtered_signals = np.zeros_like(normalized_signals)
    for i in range(signals.shape[1]):
        filtered = signal.filtfilt(b, a, normalized_signals[:, i])
        filtered_signals[:, i] = signal.filtfilt(b_notch, a_notch, filtered)
    
    # Downsample to 20 Hz for easier processing
    downsample_factor = 10  # 200Hz → 20Hz
    downsampled_signals = signal.decimate(filtered_signals, downsample_factor, axis=0)
    
    return downsampled_signals

def extract_features(signal_data, fs=20, window_size=10):
    """
    Extract features from EHG signal.
    """
    window_samples = int(window_size * fs)
    n_channels = signal_data.shape[1] if len(signal_data.shape) > 1 else 1
    if n_channels == 1:
        signal_data = signal_data.reshape(-1, 1)
    
    features = {}
    for ch in range(n_channels):
        ch_data = signal_data[:, ch]
        
        # Time domain features
        features[f'mean_ch{ch}'] = np.mean(ch_data)
        features[f'std_ch{ch}'] = np.std(ch_data)
        features[f'rms_ch{ch}'] = np.sqrt(np.mean(ch_data**2))
        features[f'kurtosis_ch{ch}'] = kurtosis(ch_data)
        features[f'skewness_ch{ch}'] = skew(ch_data)
        features[f'max_amplitude_ch{ch}'] = np.max(np.abs(ch_data))
        
        # Frequency domain features
        f, psd = signal.welch(ch_data, fs=fs, nperseg=min(window_samples, len(ch_data)))
        features[f'peak_freq_ch{ch}'] = f[np.argmax(psd)]
        features[f'energy_0.1_0.3Hz_ch{ch}'] = np.sum(psd[(f >= 0.1) & (f <= 0.3)])
        features[f'energy_0.3_1Hz_ch{ch}'] = np.sum(psd[(f >= 0.3) & (f <= 1.0)])
        features[f'energy_1_3Hz_ch{ch}'] = np.sum(psd[(f >= 1.0) & (f <= 3.0)])
        
        # Wavelet features
        coeffs = pywt.wavedec(ch_data, 'db4', level=4)
        for i, coeff in enumerate(coeffs):
            features[f'wavelet_energy_level{i}_ch{ch}'] = np.sum(coeff**2) / len(coeff)
    
    return features

def get_annotated_contractions(annotations):
    """
    Extract contraction information from annotations.
    
    Parameters:
    -----------
    annotations : wfdb.Annotation
        Annotation object from wfdb.rdann
    
    Returns:
    --------
    list
        List of contraction dictionaries with start_time, end_time, etc.
    """
    if annotations is None:
        return []
    contractions = []
    sample_times = annotations.sample / annotations.fs
    c_indices = [i for i, symbol in enumerate(annotations.symbol) if symbol in ['C', '(c)']]
    for idx in c_indices:
        start_time = sample_times[idx]
        
        # Look for the next annotation time to estimate end time
        # If this is the last annotation, use a default duration of 45 seconds
        if idx < len(sample_times) - 1:
            next_time = sample_times[idx + 1]
            # Limit maximum contraction duration to 120 seconds
            duration = min(next_time - start_time, 120)
            # If duration is too short, use a minimum of 30 seconds
            if duration < 30:
                duration = 30
        else:
            duration = 45  # Default duration for last contraction
        
        end_time = start_time + duration
        
        contractions.append({
            'start_time': start_time,
            'end_time': end_time,
            'peak_time': start_time + (duration / 2),  # Estimate peak in the middle
            'duration': duration,
            'type': annotations.symbol[idx]  # Store if it's 'C' or '(c)'
        })
    
    return contractions

def plot_signals_with_annotations(signals, annotated_contractions, fs=20, channel_idx=0):
    """
    Plot EHG signals with annotated contractions.
    """
    time = np.arange(len(signals)) / fs
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Full signal on top plot
    ax1.plot(time, signals[:, channel_idx], 'b-', alpha=0.7)
    
    # Plot annotated contractions
    for contraction in annotated_contractions:
        start = contraction['start_time']
        end = contraction['end_time']
        color = 'blue' if contraction['type'] == 'C' else 'green'
        alpha = 0.3 if contraction['type'] == 'C' else 0.2
        
        ax1.axvspan(start, end, color=color, alpha=alpha)
        ax1.axvline(start, color=color, linestyle='-', linewidth=1.5)
    
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude (µV)')
    ax1.set_title(f'EHG Signal with Annotated Contractions (Channel {channel_idx+1})')
    ax1.grid(True)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', alpha=0.3, label='Definite Contraction (C)'),
        Patch(facecolor='green', alpha=0.2, label='Possible Contraction ((c))')
    ]
    ax1.legend(handles=legend_elements, loc='upper right')
    
    # If there are contractions, zoom in on the first one
    if annotated_contractions:
        # Find the first definite contraction if available, otherwise first possible one
        definite_contractions = [c for c in annotated_contractions if c['type'] == 'C']
        first_contraction = definite_contractions[0] if definite_contractions else annotated_contractions[0]
        
        # Add padding around the contraction
        padding = 30  # seconds
        start_time = max(0, first_contraction['start_time'] - padding)
        end_time = min(len(signals)/fs, first_contraction['end_time'] + padding)
        
        start_idx = int(start_time * fs)
        end_idx = int(end_time * fs)
        
        # Zoom in on the contraction
        ax2.plot(time[start_idx:end_idx], signals[start_idx:end_idx, channel_idx], 'b-')
        
        # Highlight the contraction
        color = 'blue' if first_contraction['type'] == 'C' else 'green'
        alpha = 0.3 if first_contraction['type'] == 'C' else 0.2
        
        ax2.axvspan(first_contraction['start_time'], first_contraction['end_time'], color=color, alpha=alpha)
        ax2.axvline(first_contraction['start_time'], color=color, linestyle='-', linewidth=1.5)
        
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Amplitude (µV)')
        ax2.set_title(f'Zoomed View of Contraction (Channel {channel_idx+1})')
        ax2.grid(True)
        ax2.legend(handles=legend_elements, loc='upper right')
    else:
        # If no contractions, show a middle segment of the signal
        middle_idx = len(signals) // 2
        start_idx = max(0, middle_idx - int(60 * fs))
        end_idx = min(len(signals), middle_idx + int(60 * fs))
        
        ax2.plot(time[start_idx:end_idx], signals[start_idx:end_idx, channel_idx], 'b-')
        
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Amplitude (µV)')
        ax2.set_title(f'Zoomed View of Signal Segment (Channel {channel_idx+1})')
        ax2.grid(True)
    
    plt.tight_layout()
    return fig

def save_data(record_name, data, data_type, channel, data_dir, results_dir="../results"):
    """
    Save data to file.
    """
    out_dir = os.path.join(data_dir, results_dir)
    os.makedirs(out_dir, exist_ok=True)
    
    # For features, save as JSON
    if data_type == 'features':
        import json
        file_path = os.path.join(out_dir, f"{record_name}_features_ch{channel}.json")
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved features to {file_path}")
    
    # For contractions, save as CSV
    elif data_type == 'contractions':
        df = pd.DataFrame(data)
        file_path = os.path.join(out_dir, f"{record_name}_annotated_contractions.csv")
        df.to_csv(file_path, index=False)
        print(f"Saved contractions to {file_path}")

def main():
    parser = argparse.ArgumentParser(description='Process EHG signals and extract features from annotated contractions.')
    parser.add_argument('--record', type=str, help='Record name', required=True)
    parser.add_argument('--data_dir', type=str, default='../data', help='Directory containing the data files')
    parser.add_argument('--channels', type=str, default='all', help='Comma-separated list of channel indices to analyze (1-based); use "all" for all channels')
    parser.add_argument('--save', action='store_true', help='Whether to save results')
    
    args = parser.parse_args()
    
    start_time = time.time()

    signals, header, annotations = load_record(args.record, args.data_dir)
    if signals is None:
        return
    
    annotated_contractions = []
    if annotations is not None:
        annotated_contractions = get_annotated_contractions(annotations)
        print(f"Found {len(annotated_contractions)} annotated contractions")
        
        if args.save and annotated_contractions:
            save_data(args.record, annotated_contractions, 'contractions', None, args.data_dir)
    
    print(f"Preprocessing signals with shape {signals.shape}")
    preprocessed_signals = preprocess_signals(signals)
    print(f"Preprocessed signal shape: {preprocessed_signals.shape}")
    
    if args.channels == 'all':
        channels = list(range(preprocessed_signals.shape[1]))
    else:
        channels = [int(ch.strip()) - 1 for ch in args.channels.split(',')]
    
    for channel_idx in channels:
        if channel_idx < 0 or channel_idx >= preprocessed_signals.shape[1]:
            print(f"Channel {channel_idx + 1} is out of range (max: {preprocessed_signals.shape[1]})")
            continue
            
        print(f"\nAnalyzing channel {channel_idx + 1}")
        
        segment_size = min(60*20, len(preprocessed_signals))  # Use 60 seconds or full signal
        signal_segment = preprocessed_signals[:segment_size, :]
        
        features = extract_features(signal_segment, fs=20)
        print(f"Extracted {len(features)} features")
        
        if args.save:
            save_data(args.record, features, 'features', channel_idx+1, args.data_dir)
        
        fig = plot_signals_with_annotations(
            preprocessed_signals,
            annotated_contractions,
            fs=20,
            channel_idx=channel_idx
        )
        
        if args.save:
            out_dir = os.path.join(args.data_dir, "../results")
            os.makedirs(out_dir, exist_ok=True)
            plt.savefig(f"{out_dir}/{args.record}_signal_ch{channel_idx + 1}.png")
            plt.close(fig)
        else:
            plt.show()
    
    print(f"\nProcessing completed in {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main() 