# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 15:01:03 2021

@author: gdx
"""

import numpy as np
from sklearn.preprocessing import label_binarize
from tensorflow.keras import optimizers
from utils import mc_metrics, LossHistory
# from network import model_creat
from pre_models import model_creat

### load data
train_path = './data/dataset/mel/'
suffix = '1.npy'

x_train = np.load(train_path+'train_data_'+suffix)
y_train = np.load(train_path+'train_label_'+suffix)

x_val = np.load(train_path+'val_data_'+suffix)
y_val = np.load(train_path+'val_label_'+suffix)

num_classes = 50 #type num
x_train = x_train[:,:,:,np.newaxis].repeat(3, axis=-1) #convert to 3 channals
x_val = x_val[:,:,:,np.newaxis].repeat(3, axis=-1) #convert to 3 channals

y_train = label_binarize(y_train, list(range(num_classes)))
y_val = label_binarize(y_val, list(range(num_classes)))

x_shape = (x_train.shape[1], x_train.shape[2], 3)

### train model
batch_size = 128 #批尺寸，即一次训练所选取的样本数，batch_size=1时为在线学习
epochs = 50 #训练轮数
model_type = 'resnet50_max' # cnn, lstm
model_save_path='./model/' + model_type + '1.h5'

model = model_creat(input_shape=x_shape, output_dim=num_classes)
model.summary()

myhistory = LossHistory(model_path=model_save_path)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(lr=0.001),#'Adam',
              metrics=['categorical_accuracy'])

# #model_fit
model.fit(x_train, y_train,
          batch_size=batch_size, epochs=epochs,
          verbose=0, #进度条
          validation_data=(x_val, y_val),#验证集
          callbacks=[myhistory])

#plot acc-loss curve
myhistory.loss_plot()