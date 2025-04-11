#!/usr/bin/env python
import os
import wfdb
import numpy as np

def load_record(record_name, data_dir="../data/records"):
    """
    Loads EHG signals, header, and annotations from a PhysioNet EHGDB record.
    
    The Icelandic 16-electrode Electrohysterogram Database (EHGDB) contains
    EHG signals recorded from a 4x4 electrode grid on the abdomen of pregnant women.
    Each record contains 16 monopolar electrode signals sampled at 200 Hz.
    
    Parameters:
    -----------
    record_name : str
        Name of the record to load (e.g., 'ice027_p_5of7')
        '_p_' indicates pregnancy recordings, '_l_' indicates labor recordings
    data_dir : str
        Directory containing the record files
        
    Returns:
    --------
    signals : np.ndarray
        Array of shape (N, 16) containing the 16-channel EHG signals,
        where N is the number of samples in the recording
    header : wfdb.Record
        Header information about the record
    annotations : wfdb.Annotation or None
        Annotations for the record if available
    """
    path = os.path.join(data_dir, record_name)
    
    if not os.path.exists(path + '.hea'):
        # Try alternative paths
        alt_path = os.path.join(os.path.dirname(data_dir), 'records', record_name)
        
        if os.path.exists(alt_path + '.hea'):
            path = alt_path
        else:
            raise FileNotFoundError(f"Record {record_name} not found at {path} or {alt_path}")
    
    # Load signals and header using wfdb
    try:
        signals, header = wfdb.rdsamp(path)
    except Exception as e:
        raise
    
    # Load annotations if available
    annotations = None
    try:
        annotations = wfdb.rdann(path, 'atr')
        print(f"Successfully loaded annotations for {record_name}")
    except Exception as e:
        print(f"No annotations found for {record_name}: {e}")

    return signals, header, annotations