# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 12:59:13 2021

@author: gdx
"""

from tensorflow.keras.applications import ResNet50, VGG16, MobileNetV2
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K
from tensorflow.keras import Model

# # resnet50 = ResNet50(weights='imagenet',include_top=False,input_shape=(40,216,3))
# vgg16 = VGG16(weights='imagenet',include_top=False,input_shape=(40,216,3))
# # mobilenetv2 = MobileNetV2(weights='imagenet',include_top=False,input_shape=(40,216,3))
# for layer in vgg16.layers[-2:]:
#         layer.trainable = True
#         print(layer.name)

def model_creat(input_shape=(28, 28, 3), output_dim=8):
    # basemodel = VGG16(weights='imagenet',include_top=False,input_shape=input_shape,pooling='avg')
    basemodel = MobileNetV2(weights='imagenet',include_top=False,input_shape=input_shape,pooling='max')
    # basemodel = ResNet50(weights='imagenet',include_top=False,input_shape=input_shape,pooling='max')

    for layer in basemodel.layers:
        layer.trainable = False
    
    # for layer in vgg16.layers[-3:]:
    #     layer.trainable = True
    
    input_tensor = Input(shape=input_shape)
    x = basemodel(input_tensor)
    # x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    output_tensor = Dense(output_dim, activation = 'softmax')(x)

    model = Model(inputs=[input_tensor], outputs=[output_tensor])
    return model


if __name__ == '__main__':
    model = model_creat(input_shape=(128, 216, 3), output_dim=50)
    model.summary()
    
    # for layer in model.layers:
    #     print(layer.trainable)
    # model.save('./model_struct/cnn_atten.h5')