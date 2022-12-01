from torch.nn import ConvTranspose2d
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import ReLU
from torchvision.transforms import CenterCrop
from torch.nn import functional as F
import torch


class ConvBlock(Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = Conv2d(in_channels, out_channels, 3)
        self.relu = ReLU()
        self.conv2 = Conv2d(out_channels, out_channels, 3)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class TransposeBlock(Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.trans1 = ConvTranspose2d(in_channels, out_channels, 2, 2) #kernel_size =2 , stride = 2


    def forward(self, x):
        return self.trans1(x)


class Encoder(Module):
    def __init__(self, channels = (3,16,32,64)):
        super().__init__()
        self.encoder_blocks = ModuleList([Block(channels[i], channels[i+1]) for i in range(len(channels)-1)]) #(3, 16), (16, 32),  (32, 64)
        self.pool = MaxPool2d(2)


    def forward(self ,x):
        # initialize an empty list to store the intermediate outputs
        blockoutputs = []

        for block in self.encoder_blocks:
            x= block(x)
            blockoutputs.append(x) #outputs: (16 channel output, 32, 64channel output)
            x = self.pool(x)

        return blockoutputs




    class Decoder(Module):
        def __init__(self, channels = (64, 32, 16)):
            super().__init__()
            self.decoderblocks = ModuleList([])