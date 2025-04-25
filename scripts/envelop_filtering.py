# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 14:35:01 2023
@author: shren
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks
from scipy.interpolate import PchipInterpolator as pchip
from scipy.interpolate import CubicSpline
import glob
import math
from load_contractions import load_contraction_annotations_from_csv
from load_processed import load_processed_data
import random

def cosineSimilarity(vecA, vecB):
    dot_product = np.dot(vecA, vecB)
    norm_vec1 = np.linalg.norm(vecA)
    norm_vac2 = np.linalg.norm(vecB)
    cosine_similarity_score = dot_product / (norm_vec1 * norm_vac2)
    return cosine_similarity_score

def reactangleIndetification(vecA, vecB):
    medianA = np.median(vecA)
    medianB = np.median(vecB)
    boxA = medianA*np.ones(vecA.shape[0])
    boxB = medianB * np.ones(vecA.shape[0])
    simboxA = cosineSimilarity(vecA, boxA)
    simboxB = cosineSimilarity(vecB, boxB)
    return simboxA * simboxB

def filterPeaks(peaks_X, Peaks_Y, direction):
    aMedian = np.percentile(Peaks_Y, 70)
    fPeaks_X = []
    fPeaks_Y = []
    for i in range(len(peaks_X)):
        if direction == 0:
            if Peaks_Y[i] < aMedian:
                fPeaks_X.append(peaks_X[i])
                fPeaks_Y.append(Peaks_Y[i])
        if direction == 1:
            if Peaks_Y[i] > aMedian:
                fPeaks_X.append(peaks_X[i])
                fPeaks_Y.append(Peaks_Y[i])
    return fPeaks_X, fPeaks_Y

def filterPeaksFit(peaks_X, Peaks_Y, direction):
    aMedian = np.median(Peaks_Y)
    fPeaks_X = []
    fPeaks_Y = []
    for i in range(len(peaks_X)):
        if direction == 0:
            if Peaks_Y[i] > 1.2 * aMedian and Peaks_Y[i] != 0:
                fPeaks_X.append(peaks_X[i])
                fPeaks_Y.append(Peaks_Y[i])
        if direction == 1:
            if Peaks_Y[i] < 1.2 * aMedian and Peaks_Y[i] != 0:
                fPeaks_X.append(peaks_X[i])
                fPeaks_Y.append(Peaks_Y[i])
    return fPeaks_X, fPeaks_Y

def calculate_envelope_features(signal_data, fs=20, return_visualization=False):
    """
    Calculate envelope features from signal data.
    
    Parameters:
    -----------
    signal_data : numpy.ndarray
        Input signal data (1D array)
    fs : int, optional
        Sampling frequency (default: 20)
    return_visualization : bool, optional
        Whether to return visualization data (default: False)
    
    Returns:
    --------
    dict
        Dictionary containing envelope features
    dict (optional)
        Dictionary containing visualization data if return_visualization is True
    """
    features = {}
    vis_data = {}
    
    # signal_data should already be 1D, no need to reshape
    x = signal_data
    
    # Find peaks
    peaks_p, _ = find_peaks(x, height=0)
    peaks_p = np.concatenate(([0], peaks_p))
    peaks_n, _ = find_peaks(-x, height=0)
    peaks_n = np.concatenate(([0], peaks_n))
    
    # First stage filtering
    fPeaksX_p, fPeaksY_p = filterPeaks(peaks_p, x[peaks_p], direction=0)
    fPeaksX_n, fPeaksY_n = filterPeaks(peaks_n, x[peaks_n], direction=1)
    
    # Find points of interest (midpoints between positive and negative peaks)
    PointOfInterests = []
    for j in range(len(fPeaksX_p)):
        aPos_Peak = fPeaksX_p[j]
        distanceMinima = 1000000
        aNeg_Peak_Select = -1
        for k in range(len(fPeaksX_n)):
            aNeg_Peak = fPeaksX_n[k]
            dist = np.abs(aPos_Peak - aNeg_Peak)
            if dist < distanceMinima:
                distanceMinima = dist
                aNeg_Peak_Select = aNeg_Peak
        if distanceMinima < 10 * fs:
            PointOfInterests.append(int(np.mean([aPos_Peak, aNeg_Peak_Select])))
    
    # Second stage filtering with points of interest
    x_poi = x.copy()
    x_poi[PointOfInterests] = 0
    fPeaksX_p, fPeaksY_p = filterPeaksFit(peaks_p, x_poi[peaks_p], direction=0)
    fPeaksX_n, fPeaksY_n = filterPeaksFit(peaks_n, x_poi[peaks_n], direction=1)
    
    # Combine points of interest with filtered peaks
    peaks_p_New = np.concatenate([fPeaksX_p, np.array(PointOfInterests)])
    peaks_p_New = np.unique(peaks_p_New)
    peaks_p_New = np.sort(peaks_p_New)
    
    peaks_n_New = np.concatenate([fPeaksX_n, np.array(PointOfInterests)])
    peaks_n_New = np.unique(peaks_n_New)
    peaks_n_New = np.sort(peaks_n_New)
    
    # Create envelopes - ensure proper handling of positive and negative values
    # For upper envelope, ensure all values are non-negative
    upper_peaks = np.maximum(x[peaks_p_New], 0)  # Force positive values
    csP = pchip(peaks_p_New, upper_peaks)
    
    # For lower envelope, ensure all values are non-positive
    lower_peaks = np.minimum(x[peaks_n_New], 0)  # Force negative values
    csN = pchip(peaks_n_New, lower_peaks)
    
    xs = np.arange(0, len(x))
    upper_envelope = np.maximum(csP(xs), 0)  # Ensure upper envelope stays above zero
    lower_envelope = np.minimum(csN(xs), 0)  # Ensure lower envelope stays below zero
    
    # Calculate features
    features['envelope_upper_mean'] = np.mean(upper_envelope)
    features['envelope_upper_std'] = np.std(upper_envelope)
    features['envelope_lower_mean'] = np.mean(lower_envelope)
    features['envelope_lower_std'] = np.std(lower_envelope)
    features['envelope_range_mean'] = np.mean(upper_envelope - lower_envelope)
    features['envelope_range_std'] = np.std(upper_envelope - lower_envelope)
    features['envelope_symmetry'] = np.mean(np.abs(upper_envelope) - np.abs(lower_envelope))
    
    # Area analysis
    PositiveSignal = upper_envelope
    NegativeSignal = lower_envelope
    coeffArray = []
    cosSimArray = []
    rectIndexArray = []
    
    for i in range(len(PointOfInterests) - 1):
        sp = PointOfInterests[i]
        ep = PointOfInterests[i + 1]
        posArea = np.sum(PositiveSignal[sp:ep])
        negArea = np.sum(NegativeSignal[sp:ep])
        
        # Handle division by zero
        if posArea + negArea != 0:
            coeff = np.abs(posArea - negArea) / (posArea + negArea)
            coeffArray.append(coeff)
            
            # Calculate cosine similarity only if vectors are not all zeros
            if np.any(PositiveSignal[sp:ep]) and np.any(NegativeSignal[sp:ep]):
                cosSim = cosineSimilarity(PositiveSignal[sp:ep], NegativeSignal[sp:ep])
                cosSimDiff = cosineSimilarity(
                    pd.Series(PositiveSignal[sp:ep]).diff().values[1:],
                    pd.Series(NegativeSignal[sp:ep]).diff().values[1:]
                )
                cosSim = 0.5*(cosSim + cosSimDiff)
                cosSimArray.append(cosSim)
                
                rectIndex = reactangleIndetification(PositiveSignal[sp:ep], NegativeSignal[sp:ep])
                if not np.isnan(rectIndex):
                    rectIndexArray.append(rectIndex)
    
    # Add area-based features
    if coeffArray:
        features['area_coefficient_mean'] = np.mean(coeffArray)
        features['area_coefficient_std'] = np.std(coeffArray)
    if cosSimArray:
        features['cosine_similarity_mean'] = np.mean(cosSimArray)
        features['cosine_similarity_std'] = np.std(cosSimArray)
    if rectIndexArray:
        features['rectangle_index_mean'] = np.mean(rectIndexArray)
        features['rectangle_index_std'] = np.std(rectIndexArray)
    
    if return_visualization:
        vis_data = {
            'signal': x,
            'upper_envelope': upper_envelope,
            'lower_envelope': lower_envelope,
            'peaks_p': fPeaksX_p,
            'peaks_n': fPeaksX_n,
            'point_of_interests': PointOfInterests,
            'area_coefficients': coeffArray,
            'cosine_similarities': cosSimArray,
            'rectangle_indices': rectIndexArray
        }
        return features, vis_data
    
    return features

# Original visualization code can be kept for standalone use
if __name__ == "__main__":
    Fs = 20
    
    # Load annotations from contractions.csv
    annotations_dict = load_contraction_annotations_from_csv("contractions.csv")
    records = list(annotations_dict.keys())
    
    # Randomly select one record
    record_name = random.choice(records)
    print(f"Visualizing record: {record_name}")
    
    # Load processed data for the record
    signals, processing_info = load_processed_data(record_name, processed_dir="data/processed")
    if signals is None:
        print(f"Could not load processed data for {record_name}")
        exit(1)
        
    # Get annotations for this record
    contractions = annotations_dict[record_name]
    
    # Process each channel
    for ch in range(signals.shape[1]):
        signal = signals[:, ch]
        
        # Calculate features and visualization data
        features, vis_data = calculate_envelope_features(signal, Fs, return_visualization=True)
        
        # Create figure with two subplots
        plt.figure(figsize=(15, 10))
        
        # Full view
        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(vis_data['signal'])
        ax1.plot(vis_data['upper_envelope'])
        ax1.plot(vis_data['lower_envelope'])
        ax1.plot(np.zeros_like(vis_data['signal']), "--", color="gray")
        
        # Add contraction regions
        for contraction in contractions:
            start_idx = int(contraction['start_time'] * Fs)
            end_idx = int(contraction['end_time'] * Fs)
            if end_idx < len(vis_data['signal']):
                ax1.axvspan(start_idx, end_idx, facecolor='g', alpha=0.25)
        
        ax1.set_title(f"{record_name} - Channel {ch+1} (Full View)")
        
        # Zoomed view - show a contraction region
        if contractions:  # If there are contractions
            # Select a random contraction
            contraction = random.choice(contractions)
            start_idx = int(contraction['start_time'] * Fs)
            end_idx = int(contraction['end_time'] * Fs)
            
            # Add some padding before and after the contraction
            padding = 100  # 5 seconds at 20Hz
            zoom_start = max(0, start_idx - padding)
            zoom_end = min(len(vis_data['signal']), end_idx + padding)
            
            ax2 = plt.subplot(2, 1, 2)
            ax2.plot(vis_data['signal'][zoom_start:zoom_end])
            ax2.plot(vis_data['upper_envelope'][zoom_start:zoom_end])
            ax2.plot(vis_data['lower_envelope'][zoom_start:zoom_end])
            ax2.plot(np.zeros_like(vis_data['signal'][zoom_start:zoom_end]), "--", color="gray")
            
            # Add contraction region for zoomed view
            ax2.axvspan(start_idx - zoom_start, end_idx - zoom_start, facecolor='g', alpha=0.25)
            
            # Add vertical lines to mark contraction boundaries
            ax2.axvline(start_idx - zoom_start, color='r', linestyle='--', alpha=0.5)
            ax2.axvline(end_idx - zoom_start, color='r', linestyle='--', alpha=0.5)
            
            ax2.set_title(f"{record_name} - Channel {ch+1} (Zoomed View - Contraction Period)")
        else:
            # If no contractions, show first 1000 samples
            zoom_samples = min(1000, len(vis_data['signal']))
            ax2 = plt.subplot(2, 1, 2)
            ax2.plot(vis_data['signal'][:zoom_samples])
            ax2.plot(vis_data['upper_envelope'][:zoom_samples])
            ax2.plot(vis_data['lower_envelope'][:zoom_samples])
            ax2.plot(np.zeros_like(vis_data['signal'][:zoom_samples]), "--", color="gray")
            ax2.set_title(f"{record_name} - Channel {ch+1} (Zoomed View - First {zoom_samples/Fs:.1f} seconds)")
        
        plt.tight_layout()
        plt.show()
