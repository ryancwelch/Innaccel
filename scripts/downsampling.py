#!/usr/bin/env python
import numpy as np
from scipy import signal

def downsample_signals(signals, original_fs=200, target_fs=20):
    """
    Downsamples EHG signals to reduce data size while preserving contraction information.
    
    Since the relevant frequency content for uterine contractions lies in the 0.1-4 Hz range,
    downsampling from 200 Hz to 20 Hz preserves the signal information while 
    significantly reducing data size (90% reduction). This follows the approach
    used in Peng et al. (2019), which is appropriate for contraction detection.
    
    Parameters:
    -----------
    signals : np.ndarray
        Array of shape (N, 16) containing the 16-channel EHG signals
    original_fs : float
        Original sampling frequency (default: 200 Hz)
    target_fs : float
        Target sampling frequency (default: 20 Hz)
        
    Returns:
    --------
    downsampled_signals : np.ndarray
        Downsampled signals with shape (N/factor, 16)
    """
    decimation_factor = int(original_fs / target_fs)
    
    if decimation_factor <= 1:
        return signals  # No downsampling needed
    
    downsampled_signals = np.zeros((signals.shape[0] // decimation_factor, signals.shape[1]))
    
    for i in range(signals.shape[1]):
        downsampled_signals[:, i] = signal.decimate(signals[:, i], decimation_factor)
    
    return downsampled_signals

def downsample_with_resampling(signals, original_fs=200, target_fs=20):
    """
    Alternative downsampling method using resampling instead of decimation.
    
    This approach uses resampling which may preserve more signal characteristics
    than simple decimation but is more computationally intensive. Useful when
    more precise control over the target frequency is needed.
    
    Parameters:
    -----------
    signals : np.ndarray
        Array of shape (N, 16) containing the 16-channel EHG signals
    original_fs : float
        Original sampling frequency (default: 200 Hz)
    target_fs : float
        Target sampling frequency (default: 20 Hz)
        
    Returns:
    --------
    resampled_signals : np.ndarray
        Resampled signals
    """
    num_samples = int(signals.shape[0] * target_fs / original_fs)
    
    resampled_signals = np.zeros((num_samples, signals.shape[1]))
    for i in range(signals.shape[1]):
        resampled_signals[:, i] = signal.resample(signals[:, i], num_samples)
    
    return resampled_signals

def get_contraction_envelope(signals, fs=20, lowcut=0.01, highcut=0.1, order=4):
    """
    Extracts the low-frequency envelope of contractions.
    
    For some applications like overall contraction pattern visualization or
    long-term monitoring, we may want just the contraction envelope rather
    than the detailed signal. This applies a very low-pass filter to extract
    just the envelope.
    
    Parameters:
    -----------
    signals : np.ndarray
        Array of shape (N, 16) containing the 16-channel EHG signals
    fs : float
        Sampling frequency (default: 20 Hz)
    lowcut : float
        Lower cutoff frequency (default: 0.01 Hz)
    highcut : float
        Upper cutoff frequency (default: 0.1 Hz)
    order : int
        Filter order (default: 4)
        
    Returns:
    --------
    envelope : np.ndarray
        Contraction envelope signals
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    
    b, a = signal.butter(order, [low, high], btype='band')
    
    envelope = np.zeros_like(signals)
    for i in range(signals.shape[1]):
        filtered = signal.filtfilt(b, a, signals[:, i])
        envelope[:, i] = signal.filtfilt(b, a, np.abs(filtered))
    
    return envelope
