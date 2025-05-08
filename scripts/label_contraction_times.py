#!/usr/bin/env python
import cv2
import numpy as np
import os
from pathlib import Path
import argparse
from typing import Tuple, List, Dict
import json
from load_contractions import load_contraction_annotations_from_csv
from load_processed import load_processed_data
from scipy.stats import mode

def detect_grid_lines(image: np.ndarray, image_path: str, output_dir: str = None) -> List[int]:
    """
    Detect vertical grid lines in the tocograph image.
    Returns list of x-coordinates of grid lines.
    """
    # Convert to grayscale and preprocess
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 21, 5)
    
    # Find vertical lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, image.shape[0]//3))
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
    vertical = cv2.morphologyEx(vertical, cv2.MORPH_CLOSE, close_kernel)
    
    # Find contours of vertical lines
    contours, _ = cv2.findContours(vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get initial grid lines
    grid_lines = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h > image.shape[0] * 0.2:  # Line should span at least 20% of the height
            x_center = x + w//2
            # Avoid duplicates - use a larger threshold for initial detection
            if not any(abs(x_center - existing_x) < 10 for existing_x in grid_lines):
                grid_lines.append(x_center)
    
    # If no lines found, try Hough transform
    if not grid_lines:
        lines = cv2.HoughLinesP(vertical, 1, np.pi/180, threshold=50, 
                              minLineLength=image.shape[0]*0.2, maxLineGap=30)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(x2 - x1) < 10:  # Nearly vertical line
                    x_center = (x1 + x2) // 2
                    if not any(abs(x_center - existing_x) < 10 for existing_x in grid_lines):
                        grid_lines.append(x_center)
    
    # Sort lines and get START tick
    grid_lines = sorted(grid_lines)
    start_x, start_y = detect_start_tick(image, image_path)
    
    # Find consistent spacing if we have enough lines
    if len(grid_lines) >= 2:
        # Calculate all spacings between consecutive lines
        spacings = np.diff(grid_lines)
        
        # Find the most common spacing
        spacing, _ = mode(spacings)
        spacing = int(spacing)
        
        # Fill gaps between detected lines
        new_grid_lines = []
        for i in range(len(grid_lines) - 1):
            current_x = grid_lines[i]
            next_x = grid_lines[i + 1]
            new_grid_lines.append(current_x)
            
            # Add lines between current and next line if gap is large enough
            while current_x + spacing < next_x:
                current_x += spacing
                new_grid_lines.append(current_x)
        
        # Add the last line
        new_grid_lines.append(grid_lines[-1])
        
        # Add lines before the first detected line if needed
        first_x = new_grid_lines[0]
        while first_x - spacing > start_x:
            first_x -= spacing
            new_grid_lines.insert(0, first_x)
        
        # Add lines after the last detected line if needed
        last_x = new_grid_lines[-1]
        while last_x + spacing < image.shape[1]:
            last_x += spacing
            new_grid_lines.append(last_x)
        
        # Final deduplication step
        final_lines = []
        for x in sorted(new_grid_lines):
            if not any(abs(x - existing_x) < spacing/2 for existing_x in final_lines):
                final_lines.append(x)
        
        return final_lines
    
    # If we couldn't establish consistent spacing, use the detected lines
    return grid_lines


def calculate_time_from_grid(start_x: int, box_x: int, grid_spacing: int, time_per_grid: int = 30) -> float:
    """
    Calculate time in seconds based on grid spacing and position.
    """
    pixels_from_start = box_x - start_x
    grid_lines = pixels_from_start / grid_spacing
    return grid_lines * time_per_grid

def yolo_to_pixel_coords(yolo_coords: List[float], img_width: int, img_height: int) -> Tuple[int, int, int, int]:
    """
    Convert YOLO format coordinates (normalized) to pixel coordinates.
    Returns (x1, y1, x2, y2) in pixel coordinates.
    """
    x_center, y_center, width, height = yolo_coords
    
    # Convert to pixel coordinates
    x_center = int(x_center * img_width)
    y_center = int(y_center * img_height)
    width = int(width * img_width)
    height = int(height * img_height)
    
    # Calculate box corners
    x1 = x_center - width // 2
    y1 = y_center - height // 2
    x2 = x_center + width // 2
    y2 = y_center + height // 2
    
    return x1, y1, x2, y2

def get_ground_truth_times(record_name: str, annotations_dict: Dict, processed_dir: str) -> List[Tuple[float, float]]:
    """
    Get ground truth contraction times from annotations.
    Returns list of (start_time, end_time) tuples in seconds.
    """
    if record_name not in annotations_dict:
        return []
    # Convert annotations to times
    times = []
    for contraction in annotations_dict[record_name]:
        # Times are already in seconds in the CSV
        start_time = float(contraction['start_time'])
        end_time = float(contraction['end_time'])
        times.append((start_time, end_time))
    
    return times

def calculate_time_errors(predicted_times: List[Tuple[float, float]], 
                         ground_truth_times: List[Tuple[float, float]]) -> Dict:
    """
    Calculate error metrics between predicted and ground truth times.
    """
    if not predicted_times or not ground_truth_times:
        return None
    
    # Match predicted contractions to ground truth using Hungarian algorithm
    from scipy.optimize import linear_sum_assignment
    
    # Create cost matrix
    cost_matrix = np.zeros((len(predicted_times), len(ground_truth_times)))
    for i, (p_start, p_end) in enumerate(predicted_times):
        for j, (g_start, g_end) in enumerate(ground_truth_times):
            # Cost is sum of start and end time differences
            cost_matrix[i, j] = abs(p_start - g_start) + abs(p_end - g_end)
    
    # Find optimal matching
    pred_indices, gt_indices = linear_sum_assignment(cost_matrix)
    
    # Calculate errors for matched pairs
    start_errors = []
    end_errors = []
    duration_errors = []
    
    for pred_idx, gt_idx in zip(pred_indices, gt_indices):
        p_start, p_end = predicted_times[pred_idx]
        g_start, g_end = ground_truth_times[gt_idx]
        
        start_errors.append(abs(p_start - g_start))
        end_errors.append(abs(p_end - g_end))
        duration_errors.append(abs((p_end - p_start) - (g_end - g_start)))
    
    return {
        'mean_start_error': float(np.mean(start_errors)) if start_errors else float('inf'),
        'mean_end_error': float(np.mean(end_errors)) if end_errors else float('inf'),
        'mean_duration_error': float(np.mean(duration_errors)) if duration_errors else float('inf'),
        'matched_pairs': len(start_errors)
    }

def detect_start_tick(image: np.ndarray, image_path: str) -> Tuple[int, int]:
    """
    Detect the START tick position by reading the label file from any split.
    Returns (start_x, start_y) coordinates of the center of the START tick.
    """
    start_tick_found = False
    for split in ['train', 'val', 'test']:
        start_tick_label_path = os.path.join('data/contraction_times/start_tick_labels', split, Path(image_path).stem + '.txt')
        try:
            with open(start_tick_label_path, 'r') as f:
                start_tick_line = f.readline().strip()
                start_tick_values = [float(x) for x in start_tick_line.split()]
                start_x1, start_y1, start_x2, start_y2 = yolo_to_pixel_coords(start_tick_values[1:], image.shape[1], image.shape[0])
                start_x = (start_x1 + start_x2) // 2  # Use center of bounding box
                start_y = (start_y1 + start_y2) // 2
                start_tick_found = True
                break
        except Exception as e:
            continue
    
    if not start_tick_found:
        print(f"Warning: Could not find start tick label for {image_path} in any split")
        return None, None
        
    return start_x, start_y

def process_image(image_path: str, label_path: str, output_dir: str, 
                 annotations_dict: Dict, processed_dir: str, grid_spacing: int = 30, split: str = 'train') -> Dict:
    """
    Process a single image, detect grid lines and START tick, and label bounding boxes with times.
    Returns error metrics if ground truth is available.
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    # Read YOLO format label for contractions
    try:
        with open(label_path, 'r') as f:
            label_lines = f.readlines()
    except Exception as e:
        return None
    
    # Get START tick position
    start_x, start_y = detect_start_tick(image, image_path)
    if start_x is None:
        return None
    
    # Detect grid lines
    grid_lines = detect_grid_lines(image, image_path, output_dir)
    
    # Calculate spacing between grid lines
    if len(grid_lines) >= 2:
        spacings = np.diff(grid_lines)
        spacing = int(mode(spacings)[0])
    else:
        spacing = 30  # Default spacing if we can't detect it
    
    grid_times = []
    for i, x in enumerate(grid_lines):
        # Draw vertical grid line
        color = (0, 0, 255)  # BGR format: (Blue=0, Green=0, Red=255) for red
        cv2.line(image, (x, 0), (x, image.shape[0]), color, 1)
        
        # Calculate time based on number of lines between current line and start line
        distance_from_start = x - start_x
        time = round(distance_from_start/spacing * 30)  # 30 seconds per grid line, rounded to nearest integer
        grid_times.append(time)
        
        # Add time marker for every 5th line
        if i % 5 == 0:
            cv2.putText(image, f"{time}s", (x-10, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Draw start tick and label
    cv2.line(image, (start_x, 0), (start_x, image.shape[0]), (255, 0, 0), 2)  # Blue line
    cv2.putText(image, "START", (start_x + 5, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  # Blue text
    
    # Process each bounding box
    predicted_times = []
    for line in label_lines:
        # Parse YOLO format (class x_center y_center width height)
        values = [float(x) for x in line.strip().split()]
        if len(values) != 5:
            continue
            
        # Convert YOLO coordinates to pixel coordinates
        x1, y1, x2, y2 = yolo_to_pixel_coords(values[1:], image.shape[1], image.shape[0])
        
        left_distance_from_start = x1 - start_x
        left_time = round(left_distance_from_start/spacing * 30)  # 30 seconds per grid line, rounded to nearest integer

        right_distance_from_start = x2 - start_x
        right_time = round(right_distance_from_start/spacing * 30)  # 30 seconds per grid line, rounded to nearest integer

        # Only include times if they are non-negative
        if left_time >= 0 and right_time >= 0:
            predicted_times.append((left_time, right_time))
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add time labels
        cv2.putText(image, f"{left_time:.1f}s", (x1, y1-10), 
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(image, f"{right_time:.1f}s", (x2, y1-10), 
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw vertical lines at start and end times
        cv2.line(image, (x1, y1-20), (x1, y2+20), (0, 255, 0), 1)
        cv2.line(image, (x2, y1-20), (x2, y2+20), (0, 255, 0), 1)
    
    # Save output image
    output_path = os.path.join(output_dir, Path(image_path).name)
    cv2.imwrite(output_path, image)
    
    # Save times to text file
    times_path = os.path.join(output_dir, Path(image_path).stem + '_times.txt')
    with open(times_path, 'w') as f:
        for start_time, end_time in predicted_times:
            f.write(f"{start_time:.1f} {end_time:.1f}\n")
    
    # Get ground truth times and calculate errors
    record_name = Path(image_path).stem  # Use filename without extension to match CSV
    ground_truth_times = get_ground_truth_times(record_name, annotations_dict, processed_dir)
    print("ground truth times", ground_truth_times)
    print("predicted times", predicted_times)
    error_metrics = calculate_time_errors(predicted_times, ground_truth_times)
    
    return error_metrics

def main():
    parser = argparse.ArgumentParser(description='Label contraction times on tocograph images')
    parser.add_argument('--data_dir', type=str, default='data/contraction_times',
                      help='Directory containing the contraction_times dataset')
    parser.add_argument('--output_dir', default='data/contraction_times/results', type=str,
                      help='Directory to save labeled images and times')
    parser.add_argument('--grid_spacing', type=int, default=30,
                      help='Expected grid spacing in pixels')
    parser.add_argument('--split', type=str, default='train',
                      choices=['train', 'val', 'test'],
                      help='Which split to process')
    parser.add_argument('--annotations_csv', type=str, default='contractions.csv',
                      help='Path to CSV file with ground truth annotations')
    parser.add_argument('--processed_dir', type=str, default='data/processed',
                      help='Directory containing processed signals')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load annotations
    annotations_dict = load_contraction_annotations_from_csv(args.annotations_csv)
    
    # Set up paths
    images_dir = os.path.join(args.data_dir, 'images', args.split)
    labels_dir = os.path.join(args.data_dir, 'contraction_labels', args.split)
    
    # Process each image and collect error metrics
    all_errors = []
    for image_path in Path(images_dir).glob('*.jpg'):
        label_path = os.path.join(labels_dir, image_path.stem + '.txt')
        if os.path.exists(label_path):
            errors = process_image(str(image_path), label_path, args.output_dir,
                                 annotations_dict, args.processed_dir, args.grid_spacing, args.split)
            if errors:
                all_errors.append(errors)
                print(f"Processed {image_path.name}")
                print(f"Errors: {errors}")
    
    # Calculate and save overall statistics
    if all_errors:
        overall_stats = {
            'mean_start_error': float(np.mean([e['mean_start_error'] for e in all_errors])),
            'mean_end_error': float(np.mean([e['mean_end_error'] for e in all_errors])),
            'mean_duration_error': float(np.mean([e['mean_duration_error'] for e in all_errors])),
            'total_matched_pairs': sum(e['matched_pairs'] for e in all_errors),
            'n_images': len(all_errors)
        }
        
        with open(os.path.join(args.output_dir, 'error_stats.json'), 'w') as f:
            json.dump(overall_stats, f, indent=2)
        
        print("\nOverall Statistics:")
        print(f"Mean start time error: {overall_stats['mean_start_error']:.2f} seconds")
        print(f"Mean end time error: {overall_stats['mean_end_error']:.2f} seconds")
        print(f"Mean duration error: {overall_stats['mean_duration_error']:.2f} seconds")
        print(f"Total matched contraction pairs: {overall_stats['total_matched_pairs']}")
        print(f"Number of processed images: {overall_stats['n_images']}")

if __name__ == "__main__":
    main() 