import torch
import os
# Image paths
dataset_path = 'dataset/train'
images_path = os.path.join(dataset_path, 'images')
masks_path = os.path.join(dataset_path, 'masks')

img_size = (101,101)

classes = ['sediment', 'salt']

test_split=  0.15

device = "cuda" if torch.cuda.is_available() else "cpu"
pin_memory  = True if device == 'cuda' else False

num_channels = 1
num_classes = len(classes)
num_levels = 3 #levels in U-Net model

lr = 1e-3
num_epochs = 40
batch_size = 64

# define threshold to filter weak predictions
threshold = 0.5



#Extras

# define the path to the base output directory
BASE_OUTPUT = "output"
# define the path to the output serialized model, model training
# plot, and testing image paths
MODEL_PATH = os.path.join(BASE_OUTPUT, "unet_tgs_salt.pth")
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])
TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "test_paths.txt"])
