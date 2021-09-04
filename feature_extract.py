# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 14:30:11 2021

@author: gdx
"""

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# y, sr = librosa.load('./sound_classification_50/test/0.wav')

# plt.figure()
# librosa.display.waveplot(y, sr)
# print('Sampling Rate: ',sr,'Hz')
# print('Duration: ',len(y)/sr)
# print('Number of samples: ', len(y))

# # 
# plt.figure()
# melspec = librosa.feature.melspectrogram(y, sr, n_fft=2048, hop_length=512)
# logmelspec = librosa.power_to_db(melspec)

# librosa.display.specshow(logmelspec, sr=sr, x_axis='time', y_axis='mel')
# plt.title('melspec')

# plt.figure()
# mfcc = librosa.feature.mfcc(y, sr, dct_type=2, n_mfcc=40)
# librosa.display.specshow(mfcc, sr=sr, x_axis='time', y_axis='mel')
# plt.title('mfcc')

def mfcc_extractor(file_name):
    y, sr = librosa.load(file_name)
    mfcc = librosa.feature.mfcc(y, sr, dct_type=2, n_mfcc=40)
    return mfcc

def mel_extractor(file_name):
    y, sr = librosa.load(file_name)
    melspec = librosa.feature.melspectrogram(y, sr, n_fft=2048, hop_length=512)
    return librosa.power_to_db(melspec)


import os

path = './sound_classification_50/train/'
dirs = os.listdir(path)

mfcc_list = []
mel_list = []
label_list = []

for file_name in dirs:
    mel = mel_extractor(path+file_name)
    mfcc = mfcc_extractor(path+file_name)   
    label = file_name.split('.')[0].split('-')[-1]
    
    mfcc_list.append(mfcc)
    mel_list.append(mel)
    label_list.append(float(label))

np.save('./data/train_data_mfcc.npy', np.array(mfcc_list))
np.save('./data/train_data_mel.npy', np.array(mel_list))
np.save('./data/label.npy', np.array(label_list))
    




