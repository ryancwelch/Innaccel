# EHG Signal Analysis and Contraction Detection

This project analyzes Electrohysterogram (EHG) signals to detect uterine contractions using both traditional machine learning and deep learning approaches. The system processes raw EHG signals, extracts features, and trains models to identify contraction events.

## Dataset

The data used in this project is the [Icelandic 16-electrode Electrohysterogram Database (EHGDB)](https://physionet.org/content/ehgdb/1.0.0/) from PhysioNet, which consists of 122 16-electrode EHG recordings performed on 45 pregnant women.

When using this resource, please cite the original publication:
- Alexandersson, A., Steingrimsdottir, T., Terrien, J., Marque, C., Karlsson, B. The Icelandic 16-electrode electrohysterogram database. Sci. Data 2:150017 doi:10.1038/sdata.2015.17 (2015).

## Project Structure

```
.
├── data/                   # Directory for storing downloaded and processed data
├── models/                 # Contains trained models for contraction prediction    
├── notebooks/              # Jupyter notebooks for analysis
├── results/                # Results from analyses and models
├── scripts/                # Python scripts
├── requirements.txt        # Required Python packages
└── README.md              # This file
```

## Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/ryancwelch/Innaccel.git
   cd Innaccel
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Unix/macOS
   # or
   .\venv\Scripts\activate  # On Windows
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Data Processing

The project processes EHG signals through several stages:

1. **Signal Preprocessing** (`scripts/preprocess_ehg.py`):
   - Loads raw EHG signals
   - Removes artifacts
   - Applies bandpass filtering
   - Downsamples to 20 Hz
   - Extracts annotated contractions

2. **Feature Extraction** (`scripts/extract_features.py`):
   - Extracts time and frequency domain features
   - Generates feature matrices for machine learning

3. **Sequence Preparation** (`scripts/prepare_sequences.py`):
   - Creates sequences for LSTM training
   - Handles variable-length sequences
   - Splits data into train/test sets

## Model Training

### Traditional Machine Learning

1. **Baseline Models** (`scripts/train_and_evaluate.py`)
   - Trains and evaluates baseline ML classifiers on all datasets

2. **Baseline Models with Balancing** (`scripts/train_and_evaluate_balanced.py`)
   - Performs various class balancing techniques
   - Trains and evaluates ML classifiers on balanced datasets

### Deep Learning

1. **LSTM** (`scripts/train_lstm_sequence.py`)
   - Trains and evaluates LSTM model on sequential data

2. **CNN-LSTM** (`scripts/train_cnn_lstm_sequence.py`)
   - Trains and evaluates CNN-LSTM model on sequential data

3. **BERT** (`scripts/train_transformer_sequence.py`)
   - Trains and evaluates BERT model on sequential data

## Usage

1. **Download the Data**:
   ```bash
   python scripts/download_data.py --matlab --extract
   ```

2. **Process Signals**:
   ```bash
   python scripts/preprocess_ehg.py
   ```

3. **Extract Features**:
   ```bash
   python scripts/extract_features.py
   ```

4. **Train Models**:
   ```bash
   python scripts/train_and_evaluate.py
   python scripts/train_and_evaluate_balanced.py
   python scripts/train_lstm_sequence.py
   python scripts/train_cnn_lstm_sequence.py
   python scripts/train_transformer_sequence.py
   ```

## Results

The project generates various outputs:

- **Processed Signals**: Preprocessed EHG signals in `data/processed/`
- **Feature Matrices**: Extracted features in `data/contraction_data/`
- **Trained Models**: Saved models in `models/`
- **Visualizations**: Performance plots and signal visualizations in `results/`

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The original dataset providers: Ásgeir Alexandersson and colleagues
- PhysioNet for hosting the dataset
- MIT Laboratory for Computational Physiology for maintaining PhysioNet

## References

1. Alexandersson, A., Steingrimsdottir, T., Terrien, J., Marque, C., Karlsson, B. The Icelandic 16-electrode electrohysterogram database. Sci. Data 2:150017 doi:10.1038/sdata.2015.17 (2015).
2. Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220. 