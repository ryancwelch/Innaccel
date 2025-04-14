#!/usr/bin/env python
import numpy as np
from scipy import signal, stats

def trim_recording_edges(signals, trim_seconds=60, fs=200):
    """
    Trims the beginning and end of the recording to avoid transient effects.
    
    The beginning and end of recordings often contain artifacts from electrode
    attachment/detachment and initial movement. Following Goldsztejn et al.,
    this function removes segments from the start and end of the recording.
    
    Parameters:
    -----------
    signals : np.ndarray
        Array of shape (N, 16) containing the 16-channel EHG signals
    trim_seconds : int
        Number of seconds to trim from each end (default: 60)
    fs : float
        Sampling frequency (default: 200 Hz)
        
    Returns:
    --------
    trimmed_signals : np.ndarray
        Signals with edges trimmed
    trim_info : dict
        Information about trimming for mapping purposes:
        - start_trim_samples: number of samples trimmed from start
        - end_trim_samples: number of samples trimmed from end
        - start_time_seconds: time offset in seconds from original recording
        - end_time_seconds: time in seconds where trimmed recording ends in original
        - original_duration_seconds: duration of original recording in seconds
        - trimmed_duration_seconds: duration of trimmed recording in seconds
    """
    trim_samples = int(trim_seconds * fs)
    
    original_samples = signals.shape[0]
    original_duration_seconds = original_samples / fs
    
    if 2 * trim_samples >= signals.shape[0]:
        trim_samples = max(0, signals.shape[0] // 4)
        print(f"Warning: Trim amount too large, reduced to {trim_samples} samples")
    
    if trim_samples > 0:
        end_index = signals.shape[0] - trim_samples
        trimmed_signals = signals[trim_samples:end_index, :]
        
        start_trim_samples = trim_samples
        end_trim_samples = trim_samples
        trimmed_samples = trimmed_signals.shape[0]
    else:
        trimmed_signals = signals
        
        start_trim_samples = 0
        end_trim_samples = 0
        trimmed_samples = original_samples
    
    # Create trim information dictionary
    trim_info = {
        'start_trim_samples': start_trim_samples,
        'end_trim_samples': end_trim_samples,
        'start_time_seconds': start_trim_samples / fs,
        'end_time_seconds': (original_samples - end_trim_samples) / fs,
        'original_duration_seconds': original_duration_seconds,
        'trimmed_duration_seconds': trimmed_samples / fs,
        'original_samples': original_samples,
        'trimmed_samples': trimmed_samples,
        'sampling_rate': fs
    }
    
    return trimmed_signals, trim_info

def detect_outliers(signals, window_size=1000, threshold=3.0):
    """
    Detects outliers in the EHG signals using a moving window approach.
    
    Parameters:
    -----------
    signals : np.ndarray
        Array of shape (N, 16) containing the 16-channel EHG signals
    window_size : int
        Size of the sliding window for outlier detection (default: 1000 samples)
    threshold : float
        Z-score threshold for outlier detection (default: 3.0)
        
    Returns:
    --------
    outlier_mask : np.ndarray
        Boolean mask with True for outlier samples
    """
    window_size = min(window_size, signals.shape[0] // 2)
    
    outlier_mask = np.zeros(signals.shape, dtype=bool)
    
    for i in range(signals.shape[1]):
        channel_data = signals[:, i]
        
        ones_filter = np.ones(window_size) / window_size
        
        pad_width = window_size // 2
        padded = np.pad(channel_data, pad_width, mode='reflect')
        
        expected_output_length = len(channel_data)
        required_input_length = expected_output_length + window_size - 1
        
        if len(padded) < required_input_length:
            extra_padding = required_input_length - len(padded)
            padded = np.pad(padded, (0, extra_padding), mode='reflect')
            
        rolling_mean = np.convolve(padded, ones_filter, mode='valid')
        
        if len(rolling_mean) != len(channel_data):
            if len(rolling_mean) > len(channel_data):
                rolling_mean = rolling_mean[:len(channel_data)]
            else:
                rolling_mean = np.pad(rolling_mean, (0, len(channel_data) - len(rolling_mean)), mode='edge')
        
        sq_diff = (channel_data - rolling_mean) ** 2
        
        padded_sq = np.pad(sq_diff, pad_width, mode='reflect')
        
        if len(padded_sq) < required_input_length:
            extra_padding = required_input_length - len(padded_sq)
            padded_sq = np.pad(padded_sq, (0, extra_padding), mode='reflect')
        
        rolling_var = np.convolve(padded_sq, ones_filter, mode='valid')
        
        if len(rolling_var) != len(channel_data):
            if len(rolling_var) > len(channel_data):
                rolling_var = rolling_var[:len(channel_data)]
            else:
                rolling_var = np.pad(rolling_var, (0, len(channel_data) - len(rolling_var)), mode='edge')
        
        rolling_std = np.sqrt(rolling_var)
        
        rolling_std = np.maximum(rolling_std, 1e-10)
        
        z_scores = np.abs((channel_data - rolling_mean) / rolling_std)
        
        outlier_mask[:, i] = z_scores > threshold
    
    return outlier_mask

def remove_motion_artifacts(signals, window_size=1000, threshold=3.0, interpolate=True):
    """
    Removes motion artifacts using statistical outlier detection.
    
    Motion artifacts from maternal movements are a significant concern in EHG recordings.
    This function detects and removes these artifacts by identifying statistical outliers
    and replacing them with interpolated values.
    
    Parameters:
    -----------
    signals : np.ndarray
        Array of shape (N, 16) containing the 16-channel EHG signals
    window_size : int
        Size of the sliding window for artifact detection
    threshold : float
        Z-score threshold for outlier detection
    interpolate : bool
        Whether to interpolate over outliers (True) or replace with zeros (False)
        
    Returns:
    --------
    cleaned_signals : np.ndarray
        Signals with motion artifacts removed
    """
    cleaned_signals = signals.copy()
    outlier_mask = detect_outliers(signals, window_size, threshold)
    
    for i in range(signals.shape[1]):
        channel_outliers = outlier_mask[:, i]
        
        if np.any(channel_outliers):
            if interpolate:
                x = np.arange(len(signals))
                x_good = x[~channel_outliers]
                y_good = signals[~channel_outliers, i]
                
                    # Interpolate outliers
                cleaned_signals[channel_outliers, i] = np.interp(x[channel_outliers], x_good, y_good)
            else:
                # Replace with zeros
                cleaned_signals[channel_outliers, i] = 0
    
    return cleaned_signals

def remove_saturated_segments(signals, saturation_threshold=0.9):
    """
    Detects and removes segments where the signal is saturated.
    
    Saturation occurs when the signal reaches the maximum or minimum recordable value,
    often due to electrode issues. This function detects flat segments and replaces them.
    
    Parameters:
    -----------
    signals : np.ndarray
        Array of shape (N, 16) containing the 16-channel EHG signals
    saturation_threshold : float
        Threshold for detecting saturation (proportion of max value)
        
    Returns:
    --------
    cleaned_signals : np.ndarray
        Signals with saturated segments removed/replaced
    """
    cleaned_signals = signals.copy()
    
    for i in range(signals.shape[1]):
        channel = signals[:, i]
        
        abs_max = np.max(np.abs(channel))
        saturation_level = abs_max * saturation_threshold
        
        saturated_high = channel > saturation_level
        saturated_low = channel < -saturation_level
        saturated = saturated_high | saturated_low
        
        diff = np.diff(channel, prepend=channel[0])
        flat = np.abs(diff) < (abs_max * 0.001)
        
        saturated_segments = saturated & flat
        
        if np.any(saturated_segments):
            x = np.arange(len(channel))
            x_good = x[~saturated_segments]
            y_good = channel[~saturated_segments]
            
            cleaned_signals[saturated_segments, i] = np.interp(x[saturated_segments], x_good, y_good)
    
    return cleaned_signals

def apply_all_artifact_removal(signals, fs=200, trim_seconds=60, window_size=1000, 
                              threshold=3.0, saturation_threshold=0.9):
    """
    Applies all artifact removal steps in sequence.
    
    Parameters:
    -----------
    signals : np.ndarray
        Array of shape (N, 16) containing the 16-channel EHG signals
    fs : float
        Sampling frequency (default: 200 Hz)
    trim_seconds : int
        Number of seconds to trim from each end
    window_size : int
        Size of the sliding window for outlier detection
    threshold : float
        Z-score threshold for outlier detection
    saturation_threshold : float
        Threshold for detecting saturation
        
    Returns:
    --------
    cleaned_signals : np.ndarray
        Completely cleaned signals
    processing_info : dict
        Information about the processing steps, including trimming information
    """
    # Step 1: Trim recording edges
    trimmed_signals, trim_info = trim_recording_edges(signals, trim_seconds, fs)
    
    # Step 2: Remove saturated segments
    desaturated_signals = remove_saturated_segments(trimmed_signals, saturation_threshold)
    
    # Step 3: Remove motion artifacts
    cleaned_signals = remove_motion_artifacts(desaturated_signals, window_size, threshold)
    
    # Create processing information dictionary
    processing_info = {
        'trim_info': trim_info,
        'artifact_removal_params': {
            'window_size': window_size,
            'threshold': threshold,
            'saturation_threshold': saturation_threshold
        }
    }
    
    return cleaned_signals, processing_info
