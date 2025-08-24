import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class PTBXL_AgeDataset(Dataset):
    """
    PyTorch Dataset for PTB-XL ECG signals with age conditioning.
    
    Expects:
    - signals_path: path to .npy file with shape (N_samples, 1000, 12)
    - ages_path: path to .npy file with shape (N_samples,)
    """
    def __init__(self, signals_path, ages_path, normalize_age=True, transform=None):
        if not os.path.exists(signals_path):
            raise FileNotFoundError(f"ECG signals file not found: {signals_path}")
        if not os.path.exists(ages_path):
            raise FileNotFoundError(f"Age labels file not found: {ages_path}")
        
        self.signals = np.load(signals_path)  # shape (N, 1000, 12)
        self.ages = np.load(ages_path)        # shape (N,)
        
        if self.signals.shape[0] != self.ages.shape[0]:
            raise ValueError("Number of signals and age labels must be equal")
        
        if normalize_age:
            self.ages = (self.ages - self.ages.min()) / (self.ages.max() - self.ages.min())
        
        self.transform = transform
        
    def __len__(self):
        return len(self.signals)
    
    def __getitem__(self, idx):
        signal = self.signals[idx]   # shape (1000, 12)
        age = self.ages[idx]         # scalar
        
        if self.transform:
            signal = self.transform(signal)
        
        # PyTorch expects (channel, length)
        signal = torch.tensor(signal, dtype=torch.float32).permute(1, 0)  # (12, 1000)
        age = torch.tensor([age], dtype=torch.float32)                    # shape (1,)
        
        return signal, age

def get_data_loaders(batch_size=16, num_workers=4):
    train_dataset = PTBXL_AgeDataset("/home/any/Desktop/diffusion_models_ecg/ptbxl_dataset/ptbxl_numpy/X_train.npy", "/home/any/Desktop/diffusion_models_ecg/ptbxl_dataset/ptbxl_numpy/Y_train.npy")
    val_dataset = PTBXL_AgeDataset("/home/any/Desktop/diffusion_models_ecg/ptbxl_dataset/ptbxl_numpy/X_val.npy", "/home/any/Desktop/diffusion_models_ecg/ptbxl_dataset/ptbxl_numpy/Y_val.npy")
    test_dataset = PTBXL_AgeDataset("/home/any/Desktop/diffusion_models_ecg/ptbxl_dataset/ptbxl_numpy/X_test.npy", "/home/any/Desktop/diffusion_models_ecg/ptbxl_dataset/ptbxl_numpy/Y_test.npy")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_data_loaders(batch_size=16)

    print("Loaded dataset batches:")

    # Example check for training batch
    for i, (signals, ages) in enumerate(train_loader):
        print(f"Train Batch {i+1}:")
        print("Signals shape:", signals.shape)  # Expected: (batch_size, 12, 1000)
        print("Ages shape:", ages.shape)        # Expected: (batch_size, 1)
        break

    # Example check for validation batch
    for i, (signals, ages) in enumerate(val_loader):
        print(f"Validation Batch {i+1}:")
        print("Signals shape:", signals.shape)
        print("Ages shape:", ages.shape)
        break

    # Example check for test batch
    for i, (signals, ages) in enumerate(test_loader):
        print(f"Test Batch {i+1}:")
        print("Signals shape:", signals.shape)
        print("Ages shape:", ages.shape)
        break

