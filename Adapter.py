# -*- coding: utf-8 -*-
"""
Created on Wed May 20 20:19:35 2020

@author: joerg-de
"""


import torch
import torch.nn as nn
from torchvision import models
import random

#this methode can be used be used for all downsamplings for example in resnet

class LearnAdaptLayer(nn.Module):
    def __init__(self,Nums):
        super(LearnAdaptLayer, self).__init__()
        self.p = nn.Parameter(torch.zeros(Nums)) #todo better init
        self.m = nn.Softplus() #todo better function
        self.epsilon = 0.00001 #float min is e-45 --> max p is 9 but clamp to 8

    #only likes positive numbers but negative numbers can be used if
    #the sign is removed before the pow operations and then readded after so -1 is -1 after the operation
    def forward(self,xI):
        
        #workaround for tensorboard
        temp2 = xI.shape[2]
        temp3 = xI.shape[3]
        if not isinstance(temp2, int):
            temp2 = temp2.item()
        if not isinstance(temp3, int):
            temp3 = temp3.item()
        
        div = 1/(temp2*temp3)
        
        x = xI + self.epsilon
        
        ps = self.m(self.p) + 1
        ps = torch.clamp(ps, max=8)
        ips = 1/ps
        ps = ps.unsqueeze(1).unsqueeze(1).expand(x.shape[0],-1,x.shape[2],x.shape[3])
        ips = ips.expand(x.shape[0],-1)
        
        x = torch.pow(x, ps)
        x = torch.sum(x, dim=(2,3))
        x *= div
        x = torch.pow(x, ips)
        return x
