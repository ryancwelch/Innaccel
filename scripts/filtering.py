#!/usr/bin/env python
import numpy as np
from scipy import signal

def remove_dc_offset(signals):
    """
    Removes DC offset (mean) from each channel.
    
    Parameters:
    -----------
    signals : np.ndarray
        Array of shape (N, 16) containing the 16-channel EHG signals
        
    Returns:
    --------
    normalized_signals : np.ndarray
        Signals with DC offset removed
    """
    normalized_signals = np.zeros_like(signals)
    for i in range(signals.shape[1]):
        channel_data = signals[:, i]
        if np.isnan(channel_data).any():
            print(f"WARNING: Channel {i} contains NaNs before DC offset removal")
            valid_data = channel_data[~np.isnan(channel_data)]
            if len(valid_data) > 0:
                fill_value = np.mean(valid_data)
            else:
                fill_value = 0
            channel_data = np.nan_to_num(channel_data, nan=fill_value)
            
        normalized_signals[:, i] = channel_data - np.mean(channel_data)
    
    return normalized_signals

def apply_bandpass_filter(signals, fs=200, lowcut=0.1, highcut=4.0, order=3):
    """
    Applies a Butterworth bandpass filter to EHG signals.
    
    This filter isolates the frequency band where most uterine contraction 
    energy lies (0.1-4 Hz), while removing baseline drift and high-frequency
    noise. The range 0.2-1.2 Hz is where most contractions are observed
    according to experts on this dataset.
    
    Parameters:
    -----------
    signals : np.ndarray
        Array of shape (N, 16) containing the 16-channel EHG signals
    fs : float
        Sampling frequency (default: 200 Hz)
    lowcut : float
        Lower cutoff frequency (default: 0.1 Hz)
    highcut : float
        Upper cutoff frequency (default: 4.0 Hz)
    order : int
        Filter order (default: 3) - reduced from 5 to make filter more stable
        
    Returns:
    --------
    filtered_signals : np.ndarray
        Bandpass filtered signals
    """
    normalized_signals = remove_dc_offset(signals)
    nyquist = 0.5 * fs
    low = max(0.01, lowcut) / nyquist  # Ensure minimum of 0.01 Hz
    high = min(highcut, 0.95 * nyquist) / nyquist  # Stay below Nyquist
    
    if high <= low + 0.1:
        high = min(0.95, low + 0.1)  # Ensure at least 0.1 difference
    
    safe_order = min(order, int(signals.shape[0] / 10))  # Ensure filter order is reasonable
    safe_order = max(1, safe_order)  # Ensure minimum order of 1
    
    try:
        b, a = signal.butter(safe_order, [low, high], btype='band')
        
        filtered_signals = np.zeros_like(normalized_signals)
        for i in range(signals.shape[1]):
            channel_data = normalized_signals[:, i]
            if np.isnan(channel_data).any() or np.isinf(channel_data).any():
                print(f"WARNING: Channel {i} contains NaNs or Infs before bandpass filter")
                channel_data = np.nan_to_num(channel_data)  # Replace NaNs and Infs with 0
            
            try:
                filtered_channel = signal.filtfilt(b, a, channel_data)
                
                if np.isnan(filtered_channel).any() or np.isinf(filtered_channel).any():
                    raise ValueError("Unstable filtfilt result")
                    
                filtered_signals[:, i] = filtered_channel
            except Exception as e:
                print(f"Filtfilt failed for channel {i}: {e}. Trying lfilter instead.")
                try:
                    filtered_channel = signal.lfilter(b, a, channel_data)
                    filtered_signals[:, i] = filtered_channel
                except Exception as e2:
                    print(f"All filtering failed for channel {i}: {e2}. Using unfiltered data.")
                    filtered_signals[:, i] = channel_data  # Use original data
    except Exception as e:
        print(f"Error designing filter: {e}. Using unfiltered data.")
        return normalized_signals  # Return normalized but unfiltered signals
    
    if np.isnan(filtered_signals).any() or np.isinf(filtered_signals).any():
        print("WARNING: Bandpass filter produced NaNs or Infs. Using normalized but unfiltered data.")
        return normalized_signals
        
    return filtered_signals

def apply_notch_filter(signals, fs=200, notch_freq=50.0, q=30):
    """
    Applies a notch filter to remove power line interference.
    
    Power line interference (50 Hz in Europe/Iceland, 60 Hz in US) can 
    contaminate EHG recordings. This filter removes that specific frequency
    while preserving the rest of the signal.
    
    Parameters:
    -----------
    signals : np.ndarray
        Array of shape (N, 16) containing the 16-channel EHG signals
    fs : float
        Sampling frequency (default: 200 Hz)
    notch_freq : float
        Frequency to notch out (default: 50 Hz for European power lines)
    q : float
        Quality factor - controls notch width (default: 30)
        
    Returns:
    --------
    filtered_signals : np.ndarray
        Notch filtered signals
    """
    if notch_freq >= fs/2:
        print(f"WARNING: Notch frequency {notch_freq} Hz is too high for sampling rate {fs} Hz. Skipping notch filter.")
        return signals
    
    safe_q = min(q, 50)  # Limit Q to reasonable value
    
    try:
        b, a = signal.iirnotch(notch_freq, safe_q, fs)
        
        filtered_signals = np.zeros_like(signals)
        for i in range(signals.shape[1]):
            channel_data = signals[:, i]
            if np.isnan(channel_data).any() or np.isinf(channel_data).any():
                print(f"WARNING: Channel {i} contains NaNs or Infs before notch filter")
                channel_data = np.nan_to_num(channel_data)  # Replace NaNs and Infs with 0
            
            try:
                filtered_channel = signal.filtfilt(b, a, channel_data)
                
                if np.isnan(filtered_channel).any() or np.isinf(filtered_channel).any():
                    raise ValueError("Unstable filtfilt result")
                    
                filtered_signals[:, i] = filtered_channel
            except Exception as e:
                print(f"Filtfilt failed for channel {i}: {e}. Using unfiltered data.")
                filtered_signals[:, i] = channel_data  # Use original data
    except Exception as e:
        print(f"Error designing notch filter: {e}. Using unfiltered data.")
        return signals  # Return original signals
    
    if np.isnan(filtered_signals).any() or np.isinf(filtered_signals).any():
        print("WARNING: Notch filter produced NaNs or Infs. Using unfiltered data.")
        return signals
        
    return filtered_signals

def apply_all_filters(signals, fs=200, lowcut=0.1, highcut=4.0, 
                      bandpass_order=3, notch_freq=50.0, q=30):
    """
    Applies all filtering steps in sequence: DC offset removal, 
    notch filtering, and bandpass filtering.
    
    Parameters:
    -----------
    signals : np.ndarray
        Array of shape (N, 16) containing the 16-channel EHG signals
    fs : float
        Sampling frequency (default: 200 Hz)
    lowcut : float
        Lower cutoff frequency for bandpass (default: 0.1 Hz)
    highcut : float
        Upper cutoff frequency for bandpass (default: 4.0 Hz)
    bandpass_order : int
        Order of the bandpass filter (default: 3)
    notch_freq : float
        Frequency to notch out (default: 50 Hz)
    q : float
        Quality factor for notch filter (default: 30)
        
    Returns:
    --------
    filtered_signals : np.ndarray
        Completely filtered signals
    """
    try:
        normalized_signals = remove_dc_offset(signals)
        if np.isnan(normalized_signals).any():
            print("WARNING: DC offset removal produced NaNs. Using original signals.")
            normalized_signals = signals.copy()
        
        notch_filtered = apply_notch_filter(normalized_signals, fs, notch_freq, q)
        if np.isnan(notch_filtered).any():
            print("WARNING: Notch filter produced NaNs. Using normalized signals without notch filter.")
            notch_filtered = normalized_signals
        
        bandpass_filtered = apply_bandpass_filter(notch_filtered, fs, lowcut, highcut, bandpass_order)
        if np.isnan(bandpass_filtered).any():
            print("WARNING: Bandpass filter produced NaNs. Using signals after notch filter only.")
            bandpass_filtered = notch_filtered
        
        return bandpass_filtered
    except Exception as e:
        print(f"ERROR in filtering pipeline: {e}. Returning original signals.")
        return signals.copy()  # Return a copy of the original signals
