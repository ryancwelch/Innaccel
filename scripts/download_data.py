#!/usr/bin/env python
import os
import requests
from tqdm import tqdm
import argparse
import sys
import tarfile
import re

# Import zipfile-deflate64 instead of standard zipfile
# Add fallback to standard zipfile if zipfile-deflate64 is not available
try:
    import zipfile_deflate64 as zipfile
    print("Using zipfile_deflate64 for better compression support")
except ImportError:
    import zipfile
    print("Using standard zipfile module (may not support deflate64 compression)")

BASE_URL = "https://physionet.org/files/ehgdb/1.0.0/"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

def download_file(url, destination):
    """Download a file from a URL to a destination path with progress bar."""
    response = requests.get(url, stream=True)
    
    if response.status_code != 200:
        print(f"Failed to download {url}. Status code: {response.status_code}")
        return False
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    with open(destination, 'wb') as file, tqdm(
        desc=os.path.basename(destination),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            size = file.write(data)
            bar.update(size)
    
    return True

def download_ehgdb_index():
    info_url = BASE_URL + "info.txt"
    info_path = os.path.join(DATA_DIR, "info.txt")
    
    print(f"Downloading index file to {info_path}...")
    return download_file(info_url, info_path)

def download_matlab_package():
    matlab_url = BASE_URL + "icelandic16ehgmat.zip"
    matlab_path = os.path.join(DATA_DIR, "icelandic16ehgmat.zip")
    
    print(f"Downloading MATLAB package to {matlab_path}...")
    return download_file(matlab_url, matlab_path)

def extract_zip(zip_path, extract_to):
    """Extract a zip file with support for deflate64 compression."""
    try:
        print(f"Extracting {zip_path} to {extract_to}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Extract files one by one with progress reporting
            total_files = len(zip_ref.infolist())
            for i, file in enumerate(zip_ref.infolist()):
                zip_ref.extract(file, extract_to)
                if i % 10 == 0 or i == total_files - 1:  # Show progress every 10 files
                    print(f"Extracted {i+1}/{total_files} files ({((i+1)/total_files)*100:.1f}%)")
        return True
    except NotImplementedError as e:
        print(f"Error: {e}")
        print("The zip file uses a compression method not supported by your current Python setup.")
        print("Please install zipfile-deflate64 with: pip install zipfile-deflate64")
        return False
    except Exception as e:
        print(f"Error extracting zip file: {e}")
        return False

def download_record(record_name):
    """Download a specific record by name."""
    files_to_download = [
        f"{record_name}.atr",
        f"{record_name}.dat",
        f"{record_name}.hea",
        f"{record_name}.jpg"
    ]
    
    success = True
    for file in files_to_download:
        file_url = BASE_URL + file
        file_path = os.path.join(DATA_DIR, "records", file)
        
        print(f"Downloading {file}...")
        if not download_file(file_url, file_path):
            success = False
    
    return success

def get_all_record_names():
    info_path = os.path.join(DATA_DIR, "info.txt")
    
    # Download the index if it doesn't exist
    if not os.path.exists(info_path):
        download_ehgdb_index()
    
    # Read the index file and extract record names
    record_names = set()
    try:
        with open(info_path, 'r') as f:
            content = f.read()
            # Use regex to find all record names
            # Pattern matches strings like ice001_l, ice002_p, etc.
            matches = re.findall(r'ice\d+_[lp](?:_\d+of\d+)?', content)
            for match in matches:
                record_names.add(match)
    except Exception as e:
        print(f"Error parsing info.txt: {e}")
        return []
    
    return list(record_names)

def download_all_records():
    record_names = get_all_record_names()
    
    if not record_names:
        print("No records found in the index file.")
        return False
    
    print(f"Found {len(record_names)} records. Downloading all...")
    
    # Create records directory
    os.makedirs(os.path.join(DATA_DIR, "records"), exist_ok=True)
    
    success = True
    for record_name in record_names:
        print(f"Downloading record {record_name}...")
        if not download_record(record_name):
            print(f"Failed to download record {record_name}")
            success = False
    
    return success

def parse_arguments():
    parser = argparse.ArgumentParser(description="Download the EHGDB dataset from PhysioNet")
    parser.add_argument("--all", action="store_true", help="Download all individual record files")
    parser.add_argument("--matlab", action="store_true", help="Download the MATLAB package")
    parser.add_argument("--record", type=str, help="Download specific record (e.g., ice001_l)")
    parser.add_argument("--extract", action="store_true", help="Extract downloaded zip files")
    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Create data directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Always download the index file
    download_ehgdb_index()
    
    if args.matlab:
        matlab_path = os.path.join(DATA_DIR, "icelandic16ehgmat.zip")
        download_matlab_package()
        
        if args.extract:
            extract_zip(matlab_path, DATA_DIR)
    
    if args.all:
        print("Downloading all individual records...")
        download_all_records()
    
    if args.record:
        os.makedirs(os.path.join(DATA_DIR, "records"), exist_ok=True)
        download_record(args.record)
        
    print("Download completed successfully!")

if __name__ == "__main__":
    main() 