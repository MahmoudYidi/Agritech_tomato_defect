import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

class DayAwareAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.day_embed = nn.Embedding(7, 4)
        
        # Encoder with explicit padding
        self.encoder = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=60, padding=30),  # kernel_size//2
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=30, padding=15),  # kernel_size//2
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(2),
            nn.Flatten()
        )
        
        # Static encoder output dim calculation
        self.encoder_out_dim = 128 * 360  # 1440 / (2*2) = 360
        
        # Latent space
        self.fc1 = nn.Linear(self.encoder_out_dim + 4, 256)
        self.fc2 = nn.Linear(256, 64)
        
        # Decoder with fixed operations
        self.decoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 128*360),
            ViewLayer(128, 360),  # Replace Unflatten
            nn.Upsample(size=720),  # 360*2
            nn.Conv1d(128, 64, kernel_size=30, padding=15),
            nn.LeakyReLU(0.1),
            nn.Upsample(size=1440),
            nn.Conv1d(64, 2, kernel_size=60, padding=30),
            SliceLayer(1440)  # Ensure exact output size
        )

    def forward(self, x, days):
        ts_flat = self.encoder(x)
        day_emb = self.day_embed(days)
        z = F.leaky_relu(self.fc1(torch.cat([ts_flat, day_emb], dim=1)), 0.1)
        z = self.fc2(z)
        return self.decoder(z)

# Minimal helper layers (no complex logic)
class ViewLayer(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape
    def forward(self, x):
        return x.view(x.size(0), *self.shape)

class SliceLayer(nn.Module):
    def __init__(self, dim_size):
        super().__init__()
        self.dim_size = dim_size
    def forward(self, x):
        return x[:, :, :self.dim_size]

    

class DayDataset(Dataset):
    def __init__(self, npy_dir):
        self.files = sorted([f for f in os.listdir(npy_dir) if f.endswith('.npy')])
        self.npy_dir = npy_dir
        self.day_map = {
            'monday':0, 'tuesday':1, 'wednesday':2, 'thursday':3,
            'friday':4, 'saturday':5, 'sunday':6
        }
        self.stats = {}
        self.original_data = {}
        
        for file in self.files:
            data = np.load(os.path.join(npy_dir, file))
            self.original_data[file] = data.copy()
            self.stats[file] = {
                'temp_mean': np.nanmean(data[:,0]),
                'temp_std': np.nanstd(data[:,0]),
                'hum_mean': np.nanmean(data[:,1]),
                'hum_std': np.nanstd(data[:,1])
            }
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file = self.files[idx]
        day_name = file.split('_')[0]
        day_num = self.day_map[day_name]
        stats = self.stats[file]
        data = self.original_data[file].copy()
        
        data[:,0] = (data[:,0] - stats['temp_mean']) / (stats['temp_std'] + 1e-8)
        data[:,1] = (data[:,1] - stats['hum_mean']) / (stats['hum_std'] + 1e-8)
        data = np.nan_to_num(data, nan=0.0)
        
        return (
            torch.tensor(data.T, dtype=torch.float32),
            torch.tensor(day_num, dtype=torch.long),
            file
        )

