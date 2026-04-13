import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import os
from src.data.augmentation import TissueAugmenter

class StainingDataset(Dataset):
    """
    Dataset for paired unstained and stained pathology patches.
    """
    def __init__(self, pairs_df, transform=None):
        self.pairs = pairs_df
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        row = self.pairs.iloc[idx]
        
        # Load images
        u_img = Image.open(row['unstained']).convert('RGB')
        s_img = Image.open(row['stained']).convert('RGB')
        
        if self.transform:
            u_tensor, s_tensor = self.transform(s_img, u_img)
        else:
            # Basic normalization to [-1, 1]
            u_tensor = torch.from_numpy(np.array(u_img)).permute(2,0,1).float() / 127.5 - 1.0
            s_tensor = torch.from_numpy(np.array(s_img)).permute(2,0,1).float() / 127.5 - 1.0
            
        return u_tensor, s_tensor

def get_dataloader(pairs_df, batch_size=4, shuffle=True):
    augmenter = TissueAugmenter()
    dataset = StainingDataset(pairs_df, transform=augmenter)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)
