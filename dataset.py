from PIL import Image
from torch.utils.data import Dataset
import torch
import numpy as np
import cv2

class segmentationdataset(Dataset):
    def __init__(self, img_paths, mask_paths, transforms= None):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transforms = transforms

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.img_paths[idx]).convert("RGB"))
        mask =  np.array(Image.open(self.mask_paths[idx]).convert("L"))
        
        if self.transforms:
            image = self.transforms(image)
            mask = self.transforms(mask)
            
   
        return (image, mask)