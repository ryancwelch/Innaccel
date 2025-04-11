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