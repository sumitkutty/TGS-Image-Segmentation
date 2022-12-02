
from torchvision import transforms

from dataset import segmentationdataset
from options import *
from main_network import UNET


# Training Packages

import matplotlib.pyplot as plt
from imutils.paths import list_images
import time 
import os
from tqdm import tqdm

from sklearn.model_selection import train_test_split

#Torch
import torch
from torch.utils.data import DataLoader

from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam



def main():
    img_paths = sorted(list(list_images(images_path)))
    mask_paths = sorted(list(list_images(masks_path)))

    train_imgs,test_imgs,train_masks,test_masks = train_test_split(img_paths, 
                                                            mask_paths,test_size=test_split,
                                                            random_state = 84)
    



    # write the testing image paths to disk so that we can use then
    # when evaluating/testing our model

    os.makedirs(BASE_OUTPUT, exist_ok=True)
    if not os.path.exists(TEST_PATHS):
        print("[INFO] Saving test image paths...")
        f = open(TEST_PATHS, "w")
        f.write("\n".join(test_imgs))
        f.close()
        print("Saved Test Image Paths to Disk")

    # Initialize the transforms
    transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize(img_size),
                                    transforms.ToTensor()])

    # Initialize the dataset
    train_dataset = segmentationdataset(img_paths= train_imgs, mask_paths = train_masks, 
                                        transforms=transform)

    test_dataset = segmentationdataset(img_paths= test_imgs, mask_paths= test_masks,
                                    transforms = transform)

    # Create DataLoader
    train_DL = DataLoader(train_dataset, batch_size = batch_size, shuffle = True,num_workers=os.cpu_count())
    test_DL = DataLoader(test_dataset, batch_size = batch_size, shuffle = False, num_workers=os.cpu_count())

    #Initialize the network
    model = UNET().to(device)

    #initialize loss function
    criterion = BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr = lr)

    # Steps Per Epoch 
    train_steps_per_epoch = len(train_dataset) // batch_size
    test_steps_per_epoch = len(test_dataset) // batch_size

    #Initialize dictionary to store losses
    History = {"train_loss":[], "test_loss":[]}

    print('Training Started')
    start = time.perf_counter()
    
    # TRAINING EPOCHS ---------------------------------------------------------------->
    for e in tqdm(range(num_epochs)):
        model.train()
        total_train_loss = 0
        total_test_loss = 0
        
        
        for (i, (imgs, masks)) in enumerate(train_DL):
            imgs = imgs.to(device)
            masks = masks.to(device)
            
            preds = model(imgs)
            loss = criterion(preds, masks)
            
            if i==0:
                print(f'Batch size is : {batch_size}')
                print(f"pred shape: {preds.shape}")
                print("       Batches * Channels * H * W")
                
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss
            with torch.no_grad():
                model.eval()
                
                for (imgs, masks) in test_DL:
                    imgs = imgs.to(device)
                    masks = masks.to(device)
                    
                    preds = model(imgs)
                    total_test_loss += criterion(preds, masks)
                    
        #Here one epoch ends:
        avg_train_loss = total_train_loss/train_steps_per_epoch
        avg_test_loss = total_test_loss/test_steps_per_epoch
        
        History["train_loss"].append(avg_train_loss.cpu().detach().numpy())
        History["test_loss"].append(avg_test_loss.cpu().detach().numpy())
        
        #Progress print
        print(f"EPOCH: {i+1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.5f},  Test Loss: {avg_test_loss:.5f}")
        
    #Here all epochs done    
    total_time = time.perf_counter() - start
    print(f"TOTAL TRAINING TIME: {f'{total_time/3600:.2f} Hours' if total_time>3600 else f'{total_time/60:.2f} Mins'}")        




    # plot the losses
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(History["train_loss"], label="train_loss")
    plt.plot(History["test_loss"], label="test_loss")
    plt.title("Training Loss on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    plt.savefig(PLOT_PATH)
    # serialize the model to disk
    torch.save(model, MODEL_PATH)
    print(f"Model saved at {MODEL_PATH}")



if __name__=='__main__':
    main()
    