# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 15:25:43 2021

@author: gdx
"""

import cv2
import numpy as np 
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers
from utils import mc_metrics, LossHistory
# from network import model_creat
from pre_models import model_creat

class DataGenerator(Sequence):
    """
    基于Sequence的自定义Keras数据生成器
    """
    def __init__(self, data_path, x_y, 
                 batch_size=64, n_channels=3, dim=(155, 154), 
                 n_classes=50, shuffle=False):
        """ 初始化方法
        :param data_path: 存放图像数据的路径
        :param x_y: 图像索引+标签
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
        x_idx = self.x_y[:,0][index * self.batch_size:(index + 1) * self.batch_size].astype(np.int)
        y_idx = self.x_y[:,1][index * self.batch_size:(index + 1) * self.batch_size].astype(np.int)
        
        return self._generate_x(x_idx), self._generate_y(y_idx)        
    
    def _generate_x(self, idx):
        """生成每一批次的图像
        :param idx: 批次数据索引
        :return: 一个批次的图像
        """
        x = np.empty((self.batch_size, *self.dim, self.n_channels))
        for i in range(len(idx)):
            x[i,:] = self._load_image(self.data_path+str(idx[i])+'.jpg')
        return x
    
    def _generate_y(self, idx):
        """生成每一批次的标签
        :param idx: 批次数据的标签
        :return: 一个批次的标签
        """
        y = np.empty((self.batch_size, self.n_classes), dtype=int)
        # Generate data
        for i in range(len(idx)):
            # Store sample
            y[i,] = to_categorical(idx[i], self.n_classes)
        return y
    
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

def path_to_array(idx_path, label_path):
    """
    Parameters
    ----------
    idx_path : TYPE
        DESCRIPTION.
    label_path : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    path = '../data/jpg_data/mel_idx/'
    data_idx = np.load(path+idx_path).reshape((-1,1))
    y = np.load(path+label_path).reshape((-1,1))
    
    return np.concatenate((data_idx,y),axis=1) #x_idx, y

path_params = {'idx_path':'train_data_1.npy', 'label_path':'train_label_1.npy'}
train_x_y = path_to_array(**path_params)
path_params = {'idx_path':'val_data_1.npy', 'label_path':'val_label_1.npy'}
val_x_y = path_to_array(**path_params)


file_path = '../data/jpg_data/mel_data/'
# Parameters
params = {'dim':(100, 100),
          'n_channels': 3,
          'n_classes': 50,
          'shuffle': True}

# Generators
training_generator = DataGenerator(file_path, train_x_y, batch_size=512, **params)
validation_generator = DataGenerator(file_path, val_x_y, batch_size=160, **params)

model_save_path='./model/mobilenet.h5'
Epoch = 15

model = model_creat(input_shape=(100, 100, 3), output_dim=50)
model.summary()

myhistory = LossHistory(model_path=model_save_path)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(lr=0.001),#'Adam',
              metrics=['categorical_accuracy'])

model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    epochs=Epoch,
                    verbose=False,
                    callbacks=[myhistory])

myhistory.loss_plot()
