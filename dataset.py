import h5py
import csv
from torch.utils.data import Dataset
import torch
import numpy as np

class VCTKDEMANDDataset(Dataset):
    def __init__(self, hdf5_file, csv_file, split="train", transform=None):
        """
        Args:
            hdf5_file (str): Path to the HDF5 file.
            csv_file (str): Path to the CSV file containing metadata.
            split (str): Data split to load ('train' or 'test').
            transform (callable, optional): Optional transform to be applied
                                             on a sample.
        """
        self.hdf5_file = hdf5_file
        self.split = split
        self.transform = transform

        # Load metadata from CSV
        self.metadata = []
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['split'] == split:
                    self.metadata.append(row)

    def __len__(self):
        return len(self.metadata) // 2  # Each pair (clean, noisy) is one sample

    def __getitem__(self, idx):
        # Clean and noisy paths in the metadata
        clean_metadata = self.metadata[idx * 2]  # Clean entry
        noisy_metadata = self.metadata[idx * 2 + 1]  # Noisy entry

        # Validate consistency
        assert clean_metadata['type'] == 'clean', "Expected clean data"
        assert noisy_metadata['type'] == 'noisy', "Expected noisy data"
        assert clean_metadata['index'] == noisy_metadata['index'], "Indices mismatch"

        # Load data from HDF5
        with h5py.File(self.hdf5_file, 'r') as hdf5_file:
            clean_waveform = hdf5_file[clean_metadata['hdf5_path']][:]
            noisy_waveform = hdf5_file[noisy_metadata['hdf5_path']][:]
        
        # 如果数据是 numpy.ndarray，需要转换为 torch.Tensor
        if isinstance(clean_waveform, np.ndarray):
            clean_waveform = torch.tensor(clean_waveform, dtype=torch.float32)
        if isinstance(noisy_waveform, np.ndarray):
            noisy_waveform = torch.tensor(noisy_waveform, dtype=torch.float32)
        
        # Apply optional transforms
        if self.transform:
            clean_waveform = self.transform(clean_waveform)
            noisy_waveform = self.transform(noisy_waveform)

        return noisy_waveform, clean_waveform
