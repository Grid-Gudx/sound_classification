# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 16:30:42 2021

@author: gdx
"""

from tensorflow.keras.layers import *
import tensorflow.keras.backend as K
from tensorflow.keras import Model
from attention_module import cbam_block

def cnn_block(x, filters, kernel_size): 
    x = Conv2D(filters, kernel_size, padding='same', activation='relu')(x)
    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = Activation('relu')(x)   
    # x = BatchNormalization(axis=-1,trainable=False)(x) #对每个通道进行归一化
    x = MaxPooling2D((2,2))(x)
    return x

def res_block(x, filters, kernel_size):
    skip = Conv1D(filters, 1, padding='same')(x)
    
    x = Conv1D(filters, kernel_size, padding='same', activation='relu')(x)
    x = Conv1D(filters, kernel_size, padding='same')(x)
    
    x = add([skip, x])
    x = Activation('relu')(x)   
    # x = BatchNormalization(axis=-1,trainable=False)(x) #对每个通道进行归一化
    x = MaxPooling1D(2)(x)
    return x


def cnn(x, output_dim=8):
    x = cnn_block(x, 8, (3,3))
    x = cnn_block(x, 16, (3,3))
    x = cnn_block(x, 32, (3,3))
    x = cnn_block(x, 64, (3,3)) #out batch * width * height * channal
    
    x = cbam_block(x, ratio=8)
    
    x = GlobalAveragePooling2D()(x)
    
    # x = Lambda(lambda x: K.max(x, axis=1, keepdims=False))(x) #out batch * height * channal

    # x = Flatten()(x)
    x = Dropout(0.2)(x)   
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(output_dim, activation='softmax')(x)
    return x


def resnet(x, output_dim=8):
    x = res_block(x, 8, 3)
    x = res_block(x, 16, 3)
    x = res_block(x, 32, 3)
    x = res_block(x, 64, 3)
    
    x = Flatten()(x)
    x = Dropout(0.2)(x)   
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(output_dim, activation='softmax')(x)
    return x

def model_creat(input_shape=(28, 28, 3), output_dim=8):
    input_shape = input_shape
    input_tensor = Input(shape=input_shape)
    
    output_tensor = cnn(input_tensor, output_dim)

    model = Model(inputs=[input_tensor], outputs=[output_tensor])
    return model

if __name__ == '__main__':
    model = model_creat(input_shape=(40, 216, 1), output_dim=50)
    model.summary()
    model.save('./model_struct/cnn_atten.h5')
