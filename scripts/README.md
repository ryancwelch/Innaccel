# EHG Signal Preprocessing

## Processing Pipeline

The preprocessing pipeline consists of the following steps:

1. **Loading Data**: Load raw EHG signals from the WFDB-formatted files
2. **Artifact Removal** (optional): Remove artifacts and trim recording edges
3. **Filtering**: Apply bandpass filtering to focus on the relevant frequency bands
4. **Downsampling**: Reduce sampling rate to decrease data size while preserving important information

## Processing Modes

Two processing modes are available:

1. **Standard Processing**: Complete pipeline including artifact removal
   - Output directory: `data/processed/`

2. **NAR Processing** (No Artifact Removal): Skip the artifact removal step
   - Output directory: `data/processed_nar/`

## Usage

### Python Preprocessing Script

```bash
# Standard processing for a single record
python preprocess_ehg.py --record ice001_l_1of1 --data_dir data/records

# NAR processing for a single record
python preprocess_ehg.py --record ice001_l_1of1 --data_dir data/records --skip_artifact_removal

# Process all records with standard processing
python preprocess_ehg.py --batch --data_dir data/records

# Process all records with NAR processing
python preprocess_ehg.py --batch --data_dir data/records --skip_artifact_removal
```

### Loading Processed Data

The `load_processed.py` script provides functions for loading processed data:

```bash
# Load a specific record from the standard processed directory
python load_processed.py --record ice001_l_1of1

# Load a specific record from the NAR processed directory
python load_processed.py --record ice001_l_1of1 --processed_dir data/processed_nar

# Compare processed data from both standard and NAR processing
python load_processed.py --record ice001_l_1of1 --compare

# List all processed records in the standard directory
python load_processed.py --list_all

# List all processed records in the NAR directory
python load_processed.py --list_all --processed_dir data/processed_nar

# Load and print the processing summary
python load_processed.py --summary
```

#### Using as a Module

You can also import and use the loading functions in your own scripts:

```python
from load_processed import load_processed_data, compare_processing_methods

# Load data from standard processing
signals, info = load_processed_data('ice001_l_1of1')

# Load data from NAR processing
nar_signals, nar_info = load_processed_data('ice001_l_1of1', processed_dir='data/processed_nar')

# Compare both processing methods
(std_signals, std_info), (nar_signals, nar_info) = compare_processing_methods('ice001_l_1of1')
```

## Command Line Options

The following options are available for the preprocessing scripts:

| Option | Description | Default |
|--------|-------------|---------|
| `--record` | Record name to process | N/A (required for single mode) |
| `--data_dir` | Directory containing raw data files | `data/records` |
| `--output_dir` | Directory to save processed data | `data/processed` or `data/processed_nar` |
| `--lowcut` | Lower cutoff frequency for bandpass filter | 0.1 Hz |
| `--highcut` | Upper cutoff frequency for bandpass filter | 4.0 Hz |
| `--target_fs` | Target sampling frequency after downsampling | 20 Hz |
| `--trim_seconds` | Seconds to trim from beginning and end | 60 seconds |
| `--skip_artifact_removal` | Skip artifact removal step | False |
| `--save_intermediate` | Save intermediate results | False |
| `--batch` | Process all records in data directory | False |

## Output Files

For each processed record, the following files are generated:

1. `{record_name}_processed.npy`: Processed signal data
2. `{record_name}_processing_info.npy`: Processing metadata and parameters

If `--save_intermediate` is enabled, additional files will be saved for each processing step.

A summary file `all_processing_summary.npy` is also generated when using batch processing.

## Feature Extraction and LSTM Training

### Preparing LSTM Data

The `prepare_lstm_data.py` script extracts features from processed EHG signals and prepares sequence data for LSTM training. It works with both standard and NAR processed signals.

```bash
# Basic usage (default: 45s windows, 5s steps)
python prepare_lstm_data.py

# Use larger windows with different step size
python prepare_lstm_data.py --window_size 60 --step_size 10

# Use NAR processed signals instead of standard
python prepare_lstm_data.py --use_nar

# Process limited number of records with verbose output
python prepare_lstm_data.py --max_records 5 --verbose

# Custom directories
python prepare_lstm_data.py \
    --data_dir data/records \
    --processed_dir data/processed \
    --output_dir data/lstm_data
```

#### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--window_size` | Window size in seconds | 45 |
| `--step_size` | Step size in seconds (window stride) | 5 |
| `--data_dir` | Directory containing raw data files | `data/records` |
| `--processed_dir` | Directory containing processed signals | `data/processed` |
| `--output_dir` | Directory to save LSTM-ready data | `data/lstm_data` |
| `--use_nar` | Use NAR processed signals | False |
| `--max_records` | Maximum number of records to process | None (all) |
| `--verbose` | Print detailed processing information | False |

#### Extracted Features

For each window, the following features are extracted:

Features:
1. Time Domain Features:
   - Mean, standard deviation, RMS
   - Kurtosis, skewness
   - Maximum amplitude
   - Peak-to-peak amplitude

2. Frequency Domain Features:
   - Peak frequency and power
   - Energy in bands (0.1-0.3Hz, 0.3-1Hz, 1-3Hz)
   - Median and mean frequencies
   - Spectral edge frequencies (90%, 95%)
   - Spectral entropy

3. Propagation Features:
   - Inter-channel velocity
   - Time lag
   - Maximum correlation

4. Wavelet Features:
   - Energy at different decomposition levels

5. Cross-channel Features:
   - Mean and max coherence
   - Maximum coherence frequency

- add shape based logic

(n_records, max_seq_length, num_features)

#### Output Files

The script generates the following files in the output directory:

1. `X_sequence.npy`: Feature sequences
2. `y_sequence.npy`: Binary labels (contraction/no-contraction)
3. `config.npy`: Processing configuration and parameters

When using NAR processing, files are prefixed with 'nar_'.

### LSTM Training

After feature extraction, train the bidirectional LSTM model using:

```bash
python models/LSTM.py
```

The model architecture includes:
- Bidirectional LSTM with attention mechanism
- Two LSTM layers with dropout
- Attention layer for sequence focus
- Dense layers for classification

Training features:
- Early stopping (patience=10)
- Learning rate scheduling
- Gradient clipping
- Model checkpointing
- GPU support when available

The best model is saved as `models/best_lstm_model.pth`.

### Complete Pipeline Example

```bash
# 1. Preprocess signals (choose one)
python preprocess_ehg.py --batch  # Standard processing
python preprocess_ehg.py --batch --skip_artifact_removal  # NAR processing

# 2. Extract features with 60s windows
python prepare_lstm_data.py --window_size 60 --step_size 10 --verbose

# 3. Train LSTM model
python models/LSTM.py
```

### Tips for Better Results

- Use 45-60 second windows to capture full contractions
- Smaller step sizes (5-10s) provide more training data
- Monitor class balance in the output data
- Try both standard and NAR processed signals
- Use verbose mode to track processing progress
- Check validation loss for overfitting 