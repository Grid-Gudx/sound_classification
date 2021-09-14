# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 13:39:05 2021

@author: gdx
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchsummary import summary
import torch.nn.functional as F

class CNN(nn.Module):
    """
    input_shape: batchsize * 1 * 640
    output_shape: batchsize * num_labels
    """
    def __init__(self):
        super(CNN, self).__init__()
        self.stage1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm1d(16),
            nn.Conv1d(16, 16, 3, 1, 1),
            nn.BatchNorm1d(16),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(16, 32, 3, 1, 1),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 32, 3, 1, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.MaxPool1d(2, 2),

            nn.Conv1d(32, 64, 3, 1, 1),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 64, 3, 1, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.MaxPool1d(2, 2),

            nn.Conv1d(64, 128, 3, 1, 1),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, 3, 1, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.MaxPool1d(2, 2)
        )

    def forward(self, x):
        x=x.float()
        x = self.stage1(x)
        return x

class My_resnet18(nn.Module):
    def __init__(self, num_class):
        super(My_resnet18, self).__init__()
        resnet18 = models.resnet18(pretrained=True)
        
        resnet_layer = nn.Sequential(*list(mobilenet.children())[:-1])
        self.resnet = resnet_layer
        # print(self.resnet)
        
        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_class))

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    

class My_resnet18(nn.Module):
    def __init__(self, num_class):
        super(My_resnet18, self).__init__()
        self.conv = nn.Conv2d(1, 3, kernel_size=1)
        # resnet18 = models.resnet18(pretrained=True)
        mobilenet = models.mobilenet_v2(pretrained=True)
        
        resnet_layer = nn.Sequential(*list(mobilenet.children())[:-1])
        self.resnet = resnet_layer
        # print(self.resnet)
        
        self.amg = nn.AdaptiveMaxPool2d((1,1))
        
        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_class))

    def forward(self, x):
        x = self.resnet(x)
        x = self.amg(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
        # out = F.softmax(x,dim=1)            
        # if self.training is True: #no activation in training
        #   return x
        # else:
        #   return out


if __name__ == '__main__':

    net = My_resnet18(num_class=50)
    
    for name, param in net.named_parameters():
        # print(name)
        param.requires_grad = False
        if "fc" in name:
            param.requires_grad = True
            
    mobilenet = models.mobilenet_v2(pretrained=True)
    resnet_layer = nn.Sequential(*list(mobilenet.children())[:-1])
    print(resnet_layer)
    
    # fc_features = resnet18.classifier[1].in_features
    # resnet18.classifier[1] = nn.Linear(fc_features, 2)
    
    dummy_input = torch.randn(5,3,100,100)
    
    out = net(dummy_input)
    # out = nn.AdaptiveMaxPool2d((1,1))(out)
    print(out.shape)
    
    summary(net, input_size=(3,100,100))
    # torch.onnx.export(resnet18, dummy_input, "./model_struct/resnet18.onnx")


