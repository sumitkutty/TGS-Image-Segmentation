import torch

from network_components import *
from torch.nn import functional as F

class UNET(Module):
    def __init__(self, enc_channels = (3,16,32,64),
                 dec_channels = (64,32,16),classes = 1,
                 retain_Dim = True, outsize = (101,101)):
        super().__init__()
        
        #initialize the encoder and decoder
        self.encoder = Encoder(enc_channels)
        self.decoder = Decoder(dec_channels)
        
        # Intialize the regression head (final classno-dimensional layer)
        #This is a 1x1 convolution operation
        #in_channels = last layer's channel_no, i.e 16
        #out_channels = no of classes . i.e 1
        
        self.head = Conv2d(in_channels = dec_channels, out_channels = 1, kernel_size=1)
        self.retain_Dim = retain_Dim
        self.outsize = outsize
        
        
    def forward(self, x):
        
        encFeatures = self.encoder(x)
        #encFeatures is the blockoutputs returned by Encoder class
        
        # The final output of the encoder and the intermediate outputs from 
        #.. the back is passed
        
        decFeatures = self.decoder(encFeatures[-1], encFeatures[::-1][1:])
        
        #Output shape of decFeatures is (B, 16, H, W)
        
        final_map = self.head(decFeatures)
        
        if self.retain_Dim:
            #If true, and the output is of different shape, then interpolation 
            #..can be done to equal the size.
            final_map = F.interpolate(final_map,size = self.outsize,mode = 'bicubic')
            
        
        return final_map
    
    
        
    
    

