import argparse
import os
import h5py
import torchaudio
import csv
from tqdm import tqdm

def validate_directory(directory):
    if not os.path.exists(directory):
        raise ValueError(f"Directory '{directory}' does not exist.")
    if not os.listdir(directory):
        raise ValueError(f"Directory '{directory}' is empty.")

def create_hdf5_dataset(data_dir, output_hdf5, output_csv):
    splits = ['train', 'test']
    with h5py.File(output_hdf5, 'w') as hdf5_file, open(output_csv, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['split', 'type', 'index', 'hdf5_path', 'length', 'filename'])

        for split in splits:
            clean_dir = os.path.join(data_dir, split, 'clean')
            noisy_dir = os.path.join(data_dir, split, 'noisy')

            # Validate directories
            validate_directory(clean_dir)
            validate_directory(noisy_dir)

            clean_files = sorted(os.listdir(clean_dir))
            noisy_files = sorted(os.listdir(noisy_dir))

            # Ensure clean and noisy file counts match
            assert len(clean_files) == len(noisy_files), (
                f"Mismatch between clean and noisy files in {split}: "
                f"{len(clean_files)} clean, {len(noisy_files)} noisy"
            )

            for idx, (clean_file, noisy_file) in tqdm(enumerate(zip(clean_files, noisy_files)),
                                                      desc=f"{split} hdf5 csv making",
                                                      total=len(clean_files)):
                clean_path = os.path.join(clean_dir, clean_file)
                noisy_path = os.path.join(noisy_dir, noisy_file)

                try:
                    clean_waveform, sr_clean = torchaudio.load(clean_path)
                    noisy_waveform, sr_noisy = torchaudio.load(noisy_path)
                except Exception as e:
                    print(f"Error loading files: {clean_path}, {noisy_path}. Skipping. Error: {e}")
                    continue

                # Ensure sampling rates match
                assert sr_clean == sr_noisy, "Sampling rates of clean and noisy files do not match!"

                # Get waveform lengths
                clean_length = clean_waveform.size(1)  # Length in samples
                noisy_length = noisy_waveform.size(1)  # Length in samples

                # Create datasets in HDF5
                group = hdf5_file.create_group(f"{split}/{idx}")
                group.create_dataset("clean", data=clean_waveform.numpy())
                group.create_dataset("noisy", data=noisy_waveform.numpy())
                group.attrs["sampling_rate"] = sr_clean

                # Write index, filename, and length to CSV
                csv_writer.writerow([split, 'clean', idx, f"{split}/{idx}/clean", clean_length, clean_file])
                csv_writer.writerow([split, 'noisy', idx, f"{split}/{idx}/noisy", noisy_length, noisy_file])

    print(f"Data successfully saved to {output_hdf5} and {output_csv}")

def main():
    parser = argparse.ArgumentParser(description="Create HDF5 and CSV dataset from audio files")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to data files")
    parser.add_argument("--output_hdf5", type=str, required=True, help="Path to output HDF5 file")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to output CSV file")
    args = parser.parse_args()
    create_hdf5_dataset(args.data_dir, args.output_hdf5, args.output_csv)

if __name__ == "__main__":
    main()
