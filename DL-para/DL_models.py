# -*- coding: utf-8 -*-

import torch.nn as nn
import torch
import numpy as np

def reparameterization(mu, logvar):
    std = torch.exp(logvar / 2)
    eps = torch.randn_like(std)

    return mu + eps*std


class RDB(nn.Module):
  
    def __init__(self, filters, res_scale=0.2):
        super(RDB, self).__init__()
        self.res_scale = res_scale

        def block(in_features, non_linearity=True):
            layers = [nn.BatchNorm3d(in_features)]
            layers += [nn.ReLU(inplace=True)]
            layers += [nn.Conv3d(in_features, filters, 3, 1, 1, bias=True)]
            return nn.Sequential(*layers)

        self.b1 = block(in_features=1 * filters)
        self.b2 = block(in_features=2 * filters)
        self.b3 = block(in_features=3 * filters)
        self.b4 = block(in_features=4 * filters)
        self.b5 = block(in_features=5 * filters, non_linearity=False)
        self.blocks = [self.b1, self.b2, self.b3, self.b4, self.b5]

    def forward(self, x):
        inputs = x
        for block in self.blocks:
            out = block(inputs)
            inputs = torch.cat([inputs, out], 1)
        return out.mul(self.res_scale) + x


class MLRDB(nn.Module):
    def __init__(self, filters, res_scale=0.2):
        super(MLRDB, self).__init__()
        self.res_scale = res_scale
        self.dense_blocks = nn.Sequential(
            RDB(filters), RDB(filters), RDB(filters)
        )

    def forward(self, x):
        return self.dense_blocks(x).mul(self.res_scale) + x

class Encoder(nn.Module):
    def __init__(self, inchannels=1, outchannels=2, filters=48, num_res_blocks=1):
        super(Encoder, self).__init__()
       
        self.conv1 = nn.Conv3d(inchannels, filters, kernel_size=3, stride=2, padding=1)
        
        self.res_blocks = nn.Sequential(*[MLRDB(filters) for _ in range(num_res_blocks)])
       
        self.trans = nn.Sequential(
            nn.BatchNorm3d(filters),
            nn.ReLU(inplace=True),
            nn.Conv3d(filters, filters, kernel_size=3, stride=2, padding=1),
        )
       
        self.mu = nn.Conv3d(filters, outchannels, 3, 1, 1, bias=False)
        self.logvar = nn.Conv3d(filters, outchannels, 3, 1, 1, bias=False)

    def forward(self, img):
     
        out1 = self.conv1(img)        
        out2 = self.res_blocks(out1)   
        out3 = self.trans(out2)        

        mu, logvar = self.mu(out3), self.logvar(out3)
        z = reparameterization(mu, logvar)
        return z

    def _n_parameters(self):
        n_params = 0
        for name, param in self.named_parameters():
            n_params += param.numel()
        return n_params

class Decoder(nn.Module):
    def __init__(self, inchannels=2, outchannels=1, filters=48, num_res_blocks=1,num_upsample=2):
        super(Decoder, self).__init__()
        
        self.conv1 = nn.Conv3d(inchannels, filters, kernel_size=3, stride=1, padding=1)
       
        self.res_block1 = nn.Sequential(*[MLRDB(filters) for _ in range(num_res_blocks+1)])
        self.transup1 = nn.Sequential(
            nn.BatchNorm3d(filters),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),    
            nn.Conv3d(filters, filters, kernel_size=(4,4,3), stride=1, padding=1),             
        )
        self.res_block2 = nn.Sequential(*[MLRDB(filters) for _ in range(num_res_blocks)])
        self.transup2 = nn.Sequential(
            nn.BatchNorm3d(filters),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),  
            nn.Conv3d(filters, outchannels, kernel_size=(3,4,4), stride=1, padding=1), 
        )
        
    def forward(self, z):
        
        out1 = self.conv1(z)          
        out2 = self.res_block1(out1)   
        out = torch.add(out1, out2)   
        out3 = self.transup1(out)      
        out4 = self.res_block2(out3)   

        img = self.transup2(out4)   

        return img

    def _n_parameters(self):
        n_params= 0
        for name, param in self.named_parameters():
            n_params += param.numel()
        return n_params


class Discriminator(nn.Module):
    def __init__(self, inchannels=2, outchannels=1, filters=48):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            
            nn.Conv3d(inchannels, filters, 3, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv3d(filters, filters, 3, 1, 1, bias=True),
            nn.BatchNorm3d(filters),
            nn.LeakyReLU(0.2, inplace=True),
            
        )
                                       
        self.fc1 = nn.Sequential(     
            nn.Linear(filters * 2 * 4 * 5,128),  
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc2 = nn.Sequential(    
            nn.Linear(128, outchannels),
            nn.Sigmoid(),
        )

    def forward(self, input):
        output = self.main(input)         
        output = output.view(output.size(0), -1)       
        output1 = self.fc1(output)      
        output2 = self.fc2(output1)
      
        return output2

    def _n_parameters(self):
        n_params = 0
        for name, param in self.named_parameters():
            n_params += param.numel()
        return n_params

if __name__ == '__main__':
    encoder = Encoder()  
    decoder = Decoder()
    discriminator = Discriminator()
    print("number of parameters: {}".format(encoder._n_parameters()+decoder._n_parameters()+discriminator._n_parameters()))
