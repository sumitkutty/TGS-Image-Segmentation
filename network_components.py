from torch.nn import ConvTranspose2d
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import ReLU
from torchvision.transforms import CenterCrop

import torch


class ConvBlock(Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = Conv2d(in_channels, out_channels,kernel_size = 3)
        self.relu = ReLU()
        self.conv2 = Conv2d(out_channels, out_channels,kernel_size = 3)

    def forward(self, x):
        return self.relu(self.conv2(self.relu(self.conv1(x))))


class TransposeBlock(Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.trans1 = ConvTranspose2d(in_channels, out_channels, 
                                        kernel_size=2, stride=2) 


    def forward(self, x):
        return self.trans1(x)


class Encoder(Module):
    def __init__(self, channels = (3,16,32,64)):
        super().__init__()

        #The first level of Unet will have 16 channels, 2nd: 32 and 3rd will have 64 channels
        self.encoder_blocks = ModuleList([ConvBlock(channels[i], channels[i+1]) 
                        for i in range(len(channels)-1)]) #(3, 16), (16, 32),  (32, 64)
        self.pool = MaxPool2d(2)


    def forward(self ,x):
        # initialize an empty list to store the intermediate outputs.
        #These intermediate outputs will be passed to the decoder as a skip connec
        blockoutputs = []

        for block in self.encoder_blocks:  
            x= block(x)
            blockoutputs.append(x) #outputs: (16 channel output, 32, 64channel output)
            x = self.pool(x)

        return blockoutputs




class Decoder(Module):
    def __init__(self, channels = (64, 32, 16)):
        super().__init__()
        # (62,32), (32,14), (16)
        self.upconv_blocks = ModuleList([TransposeBlock(channels[i], 
                                channels[i+1]) for i in range(len(channels)-1)])
        #The below contains the conv blocks on the decoder side.
        self.decoder_blocks = ModuleList([ConvBlock(channels[i], 
                                channels[i+1]) for i in range(len(channels)-1)])


    def forward(self, x, encFeatures):
        for i in range(len(self.upconv)): #Goes from 0,1,2
            x = self.upconv_blocks[i](x)

            encFeat = self.crop(x , encFeatures[i])

            #Concat in dim=1. This bascially concats one row of the tensor to the other row.
            x = torch.cat([x, encFeat], dim = 1)
            x=  self.decoder_blocks[i](x)

        return x

                
    def crop(self,x, encFeat):
        """Crops the encoder feature tensor to match
            the shape of the upconvolved output

        Args:
            x (Tensor): Upconv Output Tensor
            encFeat (Tensor): Encoder Feature Tensor

        Returns:
            Tensor : Cropped(reshaped) encoder feature
        """
        (_,_, H, W) = x.shape
        encFeat = CenterCrop((H,W))(encFeat)
        return encFeat
        
    