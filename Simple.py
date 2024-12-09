from torch.utils.data import DataLoader
from dataset import VCTKDEMANDDataset

def collate_fn(batch):
    batch = [(noisy.clone().detach() if isinstance(noisy, torch.Tensor) else torch.tensor(noisy, dtype=torch.float32),
              clean.clone().detach() if isinstance(clean, torch.Tensor) else torch.tensor(clean, dtype=torch.float32))
             for noisy, clean in batch]

    max_length = max([x[0].shape[1] for x in batch])
    padded_noisy = torch.zeros(len(batch), 1, max_length)
    padded_clean = torch.zeros(len(batch), 1, max_length)
    for i, (noisy, clean) in enumerate(batch):
        length = noisy.shape[1]
        padded_noisy[i, 0, :length] = noisy
        padded_clean[i, 0, :length] = clean
    return padded_noisy, padded_clean

# 数据集加载
full_train_dataset = VCTKDEMANDDataset(
    hdf5_file=hdf5_file,
    csv_file=csv_file,
    split="train",
    transform=None
)

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
)
