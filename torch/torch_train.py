# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 14:58:41 2021

@author: gdx
"""

import numpy as np
import cv2

import torch
import time
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_net import My_resnet18

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class dataread(Dataset):
    def __init__(self, file_path, idx_path, label_path):
        self.data_idx = np.load(idx_path)
        self.y = np.load(label_path)
        self.data_path = file_path
        self.len = len(self.y)

    def _load_image(self, image_path):
       """cv2读取图像
       """
       # img = cv2.imread(image_path)
       img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
       # w, h, _ = img.shape
       # if w>h:
       #     img = np.rot90(img)
       img = cv2.resize(img, (155, 155)) / 255
       return img.astype(np.float32)
    
    def shape_transform(self,x,y):
        x=torch.from_numpy(x.transpose(2,0,1))
        return x, y
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, item):
        x = self._load_image(self.data_path+str(self.data_idx[item])+'.jpg')
        y = self.y[item]       
        return self.shape_transform(x, y)
    
def acc(_input, target):
    _input = F.softmax(_input,dim=1)
    target=target.squeeze()
    acc=float((_input.argmax(dim=1) == target).float().mean())
    return acc

def evaluate(net, val_loader, criterion):
    net.eval()
    with torch.no_grad():
        running_loss=0.
        running_acc=0.
        for i, data in enumerate(val_loader):
            inputs, labels = data
            input_val, label_val = inputs.to(device), labels.to(device)
            val_output = net(input_val)
            running_loss += criterion(val_output, label_val.long()).item()
            running_acc += acc(val_output,label_val)
    return running_acc/(i+1), running_loss/(i+1)


def train(train_data, val_data, Batch_size, Epoch, model_save_path):
    # load data
    train_loader = DataLoader(train_data, batch_size=Batch_size, shuffle=False, pin_memory=False, drop_last=False)
    val_loader = DataLoader(val_data, batch_size=160, shuffle=False, drop_last=False, pin_memory=False)
    
    net = My_resnet18(num_class=50).to(device)    
    for name, param in net.named_parameters():
        # print(name)
        param.requires_grad = False
        if "fc" in name:
            param.requires_grad = True
 
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    
    train_acc = []
    train_loss = []
    val_acc = []
    val_loss = []   
    plt.ion()
    fig = plt.figure(figsize=(10, 4))   
    start = time.process_time()
     ### the loop
    best_acc = 0.
    best_epoch = 0
    for epoch in range(Epoch):
        running_loss=0.
        running_acc=0.
        net.train()
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = Variable(inputs.to(device)), Variable(labels.to(device))
            outputs = net(inputs)
            
            optimizer.zero_grad()
            loss_value = criterion(outputs,labels.long())
            loss_value.backward()
            optimizer.step()
            
            running_loss += loss_value.item()
            running_acc += acc(outputs,labels)
        
        acc1 = running_acc/(i+1)
        loss1 = running_loss/(i+1)
        train_acc.append(acc1)
        train_loss.append(loss1)
    
        acc2, loss2 = evaluate(net, val_loader, criterion)
        val_acc.append(acc2)
        val_loss.append(loss2)
    
        if acc2 >= best_acc:
            best_acc = acc2
            best_epoch = epoch
            torch.save(net.state_dict(), model_save_path)  # 保存参数
    
        print('epoch: %d | acc: %0.5f | loss: %.5f | val_acc: %0.5f | val_loss: %.5f' % (epoch + 1, acc1,loss1,acc2,loss2))
        plt.clf()
        ax1, ax2 = fig.subplots(1, 2)
        plt.suptitle("train_history (epoch is %d, best acc is %.5f in epoch %d)"%(epoch+1, best_acc, best_epoch+1),
                     fontsize=15, y=0.99)
        ax1.set_ylabel('accuracy')
        ax1.set_xlabel('epoch')
        ax1.grid(True)
        ax1.set_title('Train and Validation accuracy')
        ax1.plot(train_acc, 'r', label='train acc')
        ax1.plot(val_acc, 'b', label='val acc')
        ax1.legend()
        
        ax2.set_ylabel('loss')
        ax2.set_xlabel('epoch')
        ax2.grid(True)
        ax2.set_title('Train and Validation loss')
        ax2.plot(train_loss, 'g', label='train loss')
        ax2.plot(val_loss, 'k', label='val loss')
        plt.legend()
        plt.pause(0.1)
        
    plt.ioff()
    end = time.process_time() 
    print('Running time is %s s'%(end-start))
    plt.show()


if __name__ == '__main__':
    file_path = '../data/jpg_data/mel_data/'
    idx_path = '../data/jpg_data/mel_idx/train_data_1.npy'
    label_path = '../data/jpg_data/mel_idx/train_label_1.npy'    
    train_data = dataread(file_path, idx_path, label_path)
    
    idx_path = '../data/jpg_data/mel_idx/val_data_1.npy'
    label_path = '../data/jpg_data/mel_idx/val_label_1.npy'
    val_data = dataread(file_path, idx_path, label_path)
    
    model_save_path = './model/mobilenetv2.pth'
    
    
    train(train_data, val_data, Batch_size=128, Epoch=50, model_save_path=model_save_path)
    


