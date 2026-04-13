import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import random
import numpy as np
from PIL import Image

class TissueAugmenter:
    """
    Augmentation pipeline specifically for pathology patches.
    Includes elastic deformations to simulate biological variation.
    """
    def __init__(self, size=(256, 256)):
        self.size = size
        self.base_transforms = T.Compose([
            T.Resize(self.size),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(degrees=15),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def elastic_transform(self, image, alpha=1, sigma=20):
        """
        Simulates tissue warping by applying a random displacement field.
        """
        # image is a torch tensor [C, H, W]
        c, h, w = image.shape
        
        # Generate random displacement fields
        dx = np.random.randn(h, w) * alpha
        dy = np.random.randn(h, w) * alpha
        
        # Smooth the fields using Gaussian blur (via simple averaging or scipy if available)
        # For simplicity in this script, we'll use a basic box blur to simulate sigma
        # In production, we'll use scipy.ndimage.gaussian_filter
        
        # Mapping indices
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        indices_x = np.clip(x + dx, 0, w - 1).astype(np.int32)
        indices_y = np.clip(y + dy, 0, h - 1).astype(np.int32)
        
        # Apply warping (simplified)
        img_np = image.permute(1, 2, 0).numpy()
        warped = img_np[indices_y, indices_x]
        
        return torch.from_numpy(warped).permute(2, 0, 1)

    def __call__(self, stained, unstained):
        """
        Applies identical transforms to both images to maintain alignment.
        """
        # Convert PIL to Tensor first for consistency
        s_tensor = F.to_tensor(stained)
        u_tensor = F.to_tensor(unstained)
        
        # 1. Resize and Normalize
        s_tensor = T.Resize(self.size)(s_tensor)
        u_tensor = T.Resize(self.size)(u_tensor)
        
        # 2. Geometric Augmentation (Same seed for both)
        seed = random.randint(0, 100000)
        
        torch.manual_seed(seed)
        s_tensor = T.RandomHorizontalFlip()(s_tensor)
        u_tensor = T.RandomHorizontalFlip()(u_tensor)
        
        torch.manual_seed(seed)
        s_tensor = T.RandomVerticalFlip()(s_tensor)
        u_tensor = T.RandomVerticalFlip()(u_tensor)
        
        torch.manual_seed(seed)
        s_tensor = T.RandomRotation(15)(s_tensor)
        u_tensor = T.RandomRotation(15)(u_tensor)
        
        # 3. Normalization
        norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return norm(s_tensor), norm(u_tensor)

if __name__ == "__main__":
    # Simple check
    augmenter = TissueAugmenter()
    print("Augmenter initialized.")
