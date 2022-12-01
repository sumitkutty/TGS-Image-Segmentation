from dataset import segmentationdataset
from options import *
from main_network import UNET


# Training Packages

import matplotlib.pyplot as plt
from imutils import paths
import time 
import os
from tqdm import tqdm

from sklearn.model_selection import train_test_split

#Torch
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam

img_paths = sorted(os.listdir(images_path))
mask_paths = sorted(os.listdir(masks_path))

train_imgs,test_imgs,train_masks,test_masks = train_test_split(img_paths, 
                                                        mask_path,test_size=test_split,
                                                        random_state = 84)




# write the testing image paths to disk so that we can use then
# when evaluating/testing our model
print("[INFO] Saving test image paths...")
f = open(TEST_PATHS, "w")
f.write("\n".join(test_imgs))
f.close()






