#!/usr/bin/env python
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
from numba import jit

@jit(nopython=True)
def cosineSimilarity(vecA, vecB):
    """
    Calculate the cosine similarity between two vectors.
    
    Cosine similarity measures the cosine of the angle between two vectors,
    providing a measure of their directional similarity regardless of magnitude.
    A value of 1 indicates identical direction, 0 indicates orthogonality,
    and -1 indicates opposite direction.
    
    Parameters:
    -----------
    vecA : numpy.ndarray
        First input vector
    vecB : numpy.ndarray
        Second input vector
        
    Returns:
    --------
    float
        Cosine similarity score between the two vectors
    """
    dot_product = np.dot(vecA, vecB)
    norm_vec1 = np.linalg.norm(vecA)
    norm_vec2 = np.linalg.norm(vecB)
    return dot_product / (norm_vec1 * norm_vec2)

@jit(nopython=True)
def reactangleIndetification(vecA, vecB):
    """
    Calculate the rectangle index between two vectors.
    
    The rectangle index measures how well two vectors can be approximated
    by rectangles (constant values). It calculates the product of cosine
    similarities between each vector and its median-based rectangle.
    
    Parameters:
    -----------
    vecA : numpy.ndarray
        First input vector
    vecB : numpy.ndarray
        Second input vector
        
    Returns:
    --------
    float
        Rectangle index score, higher values indicate more rectangular shapes
    """
    medianA = np.median(vecA)
    medianB = np.median(vecB)
    boxA = medianA * np.ones_like(vecA)
    boxB = medianB * np.ones_like(vecB)
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

def calculate_envelope_features(
    signal_data, 
    fs=20, 
    return_visualization=False,
    # Hyperparameters for peak detection
    peak_height_threshold=0,
    peak_distance=None,
    # Hyperparameters for first stage filtering
    first_stage_percentile=70,
    # Hyperparameters for second stage filtering
    second_stage_multiplier=1.2,
    # Hyperparameters for point of interest detection
    poi_distance_threshold=10,
    # Hyperparameters for area analysis
    min_segment_length=5
):
    """
    Calculate envelope features from signal data with configurable parameters.
    
    This function performs the following steps:
    1. Peak Detection: Identifies local maxima and minima in the signal
    2. First Stage Filtering: Removes peaks below/above percentile thresholds
    3. Point of Interest Detection: Finds midpoints between positive and negative peaks
    4. Second Stage Filtering: Further refines peaks using median-based thresholds
    5. Envelope Creation: Constructs upper and lower envelopes using PCHIP interpolation
    6. Feature Extraction: Calculates statistical features from the envelopes
    
    The function is optimized for performance and allows tuning of key parameters
    for different signal characteristics.
    
    Parameters:
    -----------
    signal_data : numpy.ndarray
        Input signal data (1D array)
    fs : int, optional
        Sampling frequency in Hz (default: 20)
    return_visualization : bool, optional
        Whether to return visualization data (default: False)
    peak_height_threshold : float, optional
        Minimum height of peaks to consider (default: 0)
    peak_distance : int, optional
        Minimum distance between peaks in samples (default: None)
    first_stage_percentile : float, optional
        Percentile threshold for first stage filtering (default: 70)
    second_stage_multiplier : float, optional
        Multiplier for second stage filtering threshold (default: 1.2)
    poi_distance_threshold : int, optional
        Maximum distance between peaks to consider as points of interest (default: 10)
    min_segment_length : int, optional
        Minimum length of segments for area analysis (default: 5)
    
    Returns:
    --------
    dict
        Dictionary containing the following envelope features:
        - envelope_upper_mean: Mean of upper envelope
        - envelope_upper_std: Standard deviation of upper envelope
        - envelope_lower_mean: Mean of lower envelope
        - envelope_lower_std: Standard deviation of lower envelope
        - envelope_range_mean: Mean of envelope range
        - envelope_range_std: Standard deviation of envelope range
        - envelope_symmetry: Symmetry measure between upper and lower envelopes
        - area_coefficient_mean: Mean of area coefficients (0 if no valid segments)
        - area_coefficient_std: Standard deviation of area coefficients (0 if no valid segments)
        - cosine_similarity_mean: Mean of cosine similarities (0 if no valid segments)
        - cosine_similarity_std: Standard deviation of cosine similarities (0 if no valid segments)
        - rectangle_index_mean: Mean of rectangle indices (0 if no valid segments)
        - rectangle_index_std: Standard deviation of rectangle indices (0 if no valid segments)
    
    dict (optional)
        Dictionary containing visualization data if return_visualization is True:
        - signal: Original signal
        - upper_envelope: Upper envelope
        - lower_envelope: Lower envelope
        - peaks_p: Positive peak locations
        - peaks_n: Negative peak locations
        - point_of_interests: Points of interest locations
    """
    features = {}
    vis_data = {}
    
    # Initialize area-based features to 0
    features['area_coefficient_mean'] = 0.0
    features['area_coefficient_std'] = 0.0
    features['cosine_similarity_mean'] = 0.0
    features['cosine_similarity_std'] = 0.0
    features['rectangle_index_mean'] = 0.0
    features['rectangle_index_std'] = 0.0
    
    # Ensure signal is 1D
    x = signal_data.flatten()
    
    # Find peaks using vectorized operations with configurable parameters
    peaks_p, _ = find_peaks(x, height=peak_height_threshold, distance=peak_distance)
    peaks_n, _ = find_peaks(-x, height=peak_height_threshold, distance=peak_distance)
    
    # Add start point if not already included
    if peaks_p[0] != 0:
        peaks_p = np.concatenate(([0], peaks_p))
    if peaks_n[0] != 0:
        peaks_n = np.concatenate(([0], peaks_n))
    
    # First stage filtering - vectorized with configurable percentile
    median_p = np.percentile(x[peaks_p], first_stage_percentile)
    median_n = np.percentile(x[peaks_n], first_stage_percentile)
    
    mask_p = x[peaks_p] < median_p
    mask_n = x[peaks_n] > median_n
    
    fPeaksX_p = peaks_p[mask_p]
    fPeaksY_p = x[fPeaksX_p]
    fPeaksX_n = peaks_n[mask_n]
    fPeaksY_n = x[fPeaksX_n]
    
    # Find points of interest using vectorized operations with configurable threshold
    if len(fPeaksX_p) > 0 and len(fPeaksX_n) > 0:
        # Create distance matrix between positive and negative peaks
        dist_matrix = np.abs(fPeaksX_p[:, np.newaxis] - fPeaksX_n)
        
        # Find closest negative peak for each positive peak
        min_dist_idx = np.argmin(dist_matrix, axis=1)
        min_dist = np.min(dist_matrix, axis=1)
        
        # Filter based on configurable distance threshold
        valid_mask = min_dist < poi_distance_threshold * fs
        PointOfInterests = np.mean([fPeaksX_p[valid_mask], 
                                  fPeaksX_n[min_dist_idx[valid_mask]]], 
                                 axis=0).astype(int)
    else:
        PointOfInterests = np.array([], dtype=int)
    
    # Second stage filtering - vectorized with configurable multiplier
    x_poi = x.copy()
    x_poi[PointOfInterests] = 0
    
    median_p = np.median(x_poi[peaks_p])
    median_n = np.median(x_poi[peaks_n])
    
    mask_p = x_poi[peaks_p] > second_stage_multiplier * median_p
    mask_n = x_poi[peaks_n] < second_stage_multiplier * median_n
    
    fPeaksX_p = peaks_p[mask_p]
    fPeaksY_p = x_poi[fPeaksX_p]
    fPeaksX_n = peaks_n[mask_n]
    fPeaksY_n = x_poi[fPeaksX_n]
    
    # Combine points of interest with filtered peaks
    peaks_p_New = np.unique(np.concatenate([fPeaksX_p, PointOfInterests]))
    peaks_n_New = np.unique(np.concatenate([fPeaksX_n, PointOfInterests]))
    
    # Create envelopes using vectorized operations
    upper_peaks = np.maximum(x[peaks_p_New], 0)
    lower_peaks = np.minimum(x[peaks_n_New], 0)
    
    csP = pchip(peaks_p_New, upper_peaks)
    csN = pchip(peaks_n_New, lower_peaks)
    
    xs = np.arange(len(x))
    upper_envelope = np.maximum(csP(xs), 0)
    lower_envelope = np.minimum(csN(xs), 0)
    
    # Calculate basic envelope features
    features['envelope_upper_mean'] = np.mean(upper_envelope)
    features['envelope_upper_std'] = np.std(upper_envelope)
    features['envelope_lower_mean'] = np.mean(lower_envelope)
    features['envelope_lower_std'] = np.std(lower_envelope)
    features['envelope_range_mean'] = np.mean(upper_envelope - lower_envelope)
    features['envelope_range_std'] = np.std(upper_envelope - lower_envelope)
    features['envelope_symmetry'] = np.mean(np.abs(upper_envelope) - np.abs(lower_envelope))
    
    # Area analysis - vectorized with configurable minimum segment length
    if len(PointOfInterests) > 1:
        # Calculate areas between consecutive points of interest
        start_points = PointOfInterests[:-1]
        end_points = PointOfInterests[1:]
        
        # Filter segments by minimum length
        segment_lengths = end_points - start_points
        valid_segments = segment_lengths >= min_segment_length
        
        if np.any(valid_segments):
            start_points = start_points[valid_segments]
            end_points = end_points[valid_segments]
            
            # Calculate areas using vectorized operations
            pos_areas = np.array([np.sum(upper_envelope[sp:ep]) for sp, ep in zip(start_points, end_points)])
            neg_areas = np.array([np.sum(lower_envelope[sp:ep]) for sp, ep in zip(start_points, end_points)])
            
            # Calculate coefficients
            total_areas = pos_areas + neg_areas
            valid_mask = total_areas != 0
            if np.any(valid_mask):
                coeffArray = np.abs(pos_areas[valid_mask] - neg_areas[valid_mask]) / total_areas[valid_mask]
                features['area_coefficient_mean'] = np.mean(coeffArray)
                features['area_coefficient_std'] = np.std(coeffArray)
            
            # Calculate cosine similarities and rectangle indices
            cosSimArray = []
            rectIndexArray = []
            
            for sp, ep in zip(start_points[valid_mask], end_points[valid_mask]):
                pos_segment = upper_envelope[sp:ep]
                neg_segment = lower_envelope[sp:ep]
                
                if np.any(pos_segment) and np.any(neg_segment):
                    cosSim = cosineSimilarity(pos_segment, neg_segment)
                    cosSimDiff = cosineSimilarity(
                        np.diff(pos_segment),
                        np.diff(neg_segment)
                    )
                    cosSim = 0.5 * (cosSim + cosSimDiff)
                    cosSimArray.append(cosSim)
                    
                    rectIndex = reactangleIndetification(pos_segment, neg_segment)
                    if not np.isnan(rectIndex):
                        rectIndexArray.append(rectIndex)
            
            # Update area-based features if we have valid values
            if len(cosSimArray) > 0:
                features['cosine_similarity_mean'] = np.mean(cosSimArray)
                features['cosine_similarity_std'] = np.std(cosSimArray)
            if len(rectIndexArray) > 0:
                features['rectangle_index_mean'] = np.mean(rectIndexArray)
                features['rectangle_index_std'] = np.std(rectIndexArray)
    
    if return_visualization:
        vis_data = {
            'signal': x,
            'upper_envelope': upper_envelope,
            'lower_envelope': lower_envelope,
            'peaks_p': fPeaksX_p,
            'peaks_n': fPeaksX_n,
            'point_of_interests': PointOfInterests
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
