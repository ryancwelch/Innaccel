# EHG Contraction Prediction Project

This project aims to develop methods for predicting contractions in pregnant women using electrohysterogram (EHG) data from the Icelandic 16-electrode Electrohysterogram Database available on PhysioNet.

## Dataset

The data used in this project is the [Icelandic 16-electrode Electrohysterogram Database (EHGDB)](https://physionet.org/content/ehgdb/1.0.0/) from PhysioNet, which consists of 122 16-electrode EHG recordings performed on 45 pregnant women.

When using this resource, please cite the original publication:
- Alexandersson, A., Steingrimsdottir, T., Terrien, J., Marque, C., Karlsson, B. The Icelandic 16-electrode electrohysterogram database. Sci. Data 2:150017 doi:10.1038/sdata.2015.17 (2015).

## Project Structure

```
.
├── data/                   # Directory for storing downloaded data
├── models/                 # Contains models for contraction prediction    
├── notebooks/              # Jupyter notebooks for analysis
├── results/                # Results from analyses and models
├── scripts/                # Python scripts
├── requirements.txt        # Required Python packages
└── README.md               # This file
```

## Getting Started

### Prerequisites


### Installation

1. Clone this repository:
   ```
   git clone https://github.com/ryancwelch/Innaccel.git
   cd Innaccel
   ```

2. Create and activate a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate
   ```

3. Install required packages:
   ```
   pip install -r requirements.txt
   ```

### Downloading the Data

Use the provided script to download data from PhysioNet:

```
python scripts/download_data.py --matlab --extract
```
<!-- 
### Running the Analysis

1. Start Jupyter Notebook:
   ```
   jupyter notebook
   ```

2. Open the `notebooks/contraction_prediction.ipynb` notebook and follow the step-by-step analysis. -->

## Processing EHG Data

The `process_ehg.py` script provides functions for:

1. Loading EHG records
2. Preprocessing and filtering signals
3. Detecting contractions
4. Extracting features for prediction
5. Visualizing signals and contractions

<!-- Example usage:

```
python scripts/process_ehg.py --record ice001_l --channel 0 --save
``` -->

This will process the record, detect contractions, and save the results to the `results/` directory.

<!-- ## Building Prediction Models

The Jupyter notebook contains examples of how to:

1. Extract features from EHG signals
2. Build machine learning models for contraction prediction
3. Evaluate model performance
4. Visualize results

## License

This project is licensed under the MIT License - see the LICENSE file for details. -->

<!-- ## Acknowledgments

- The original dataset providers: Ásgeir Alexandersson and colleagues
- PhysioNet for hosting the dataset
- MIT Laboratory for Computational Physiology for maintaining PhysioNet -->

## References

1. Alexandersson, A., Steingrimsdottir, T., Terrien, J., Marque, C., Karlsson, B. The Icelandic 16-electrode electrohysterogram database. Sci. Data 2:150017 doi:10.1038/sdata.2015.17 (2015).
2. Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220. 