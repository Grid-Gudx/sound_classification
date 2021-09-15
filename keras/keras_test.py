# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 14:17:09 2021

@author: gdx
"""

import sys
sys.path.append('../')

import cv2
import numpy as np 
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import load_model

class DataGenerator(Sequence):
    """
    基于Sequence的自定义Keras数据生成器
    """
    def __init__(self, data_path, x_y, 
                 batch_size=64, n_channels=3, dim=(155, 154), 
                 n_classes=50, shuffle=False):
        """ 初始化方法
        :param data_path: 存放图像数据的路径
        :param x_y: 图像索引
        :param batch_size: batch size 
        :param dim: 图像大小
        :param n_channels: 图像通道
        :param n_classes: 标签类别数
        :param shuffle: 是否先打乱数据
        """
        self.data_path = data_path        
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.dim = dim
        self.n_classes = n_classes
        if shuffle:
            np.random.shuffle(x_y)
        self.x_y = x_y
    
    def __len__(self):
        """每个epoch下的批次数量
        """
        return int(self.x_y.shape[0] // self.batch_size)
    
    def __getitem__(self, index):
        """生成每一批次训练数据
        :param index: 批次索引
        :return: 训练图像和标签
        """
        # 生成批次索引
        x_idx = self.x_y[index * self.batch_size:(index + 1) * self.batch_size].astype(np.int)
        
        return self._generate_x(x_idx)   
    
    def _generate_x(self, idx):
        """生成每一批次的图像
        :param idx: 批次数据索引
        :return: 一个批次的图像
        """
        x = np.empty((self.batch_size, *self.dim, self.n_channels))
        for i in range(len(idx)):
            x[i,:] = self._load_image(self.data_path+str(idx[i])+'.jpg')
        return x
    
    def _load_image(self, image_path):
       """cv2读取图像
       """
       # img = cv2.imread(image_path)
       img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
       # w, h, _ = img.shape
       # if w>h:
       #     img = np.rot90(img)
       img = cv2.resize(img, (self.dim[1],self.dim[0])) / 255 #Tip: width*height
       return img.astype(np.float32)


### load test_data
x_y = np.array(range(400))

file_path = '../data/jpg_data/test_mel_data/'
# Parameters
params = {'dim':(150, 150),
          'n_channels': 3,
          'n_classes': 50,
          'shuffle': False}

# Generators
training_generator = DataGenerator(file_path, x_y, batch_size=400, **params)

test_x = training_generator.__getitem__(0)

### load model
model_save_path='./model/mobilenet_1.h5'
model = load_model(model_save_path, compile=False)
y_score = model.predict(test_x)
y_pred = np.argmax(y_score, axis=1)

### save result
import pandas as pd 

a = np.concatenate((x_y.reshape(-1,1),y_pred.reshape(-1,1)),axis=1)
pd.DataFrame(a).to_csv('./result/sample.csv',index=None,header=False)


