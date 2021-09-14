# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 14:30:11 2021

@author: gdx
"""

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

### show the wave example
# def wave_show(y, sr):
#     print('Sampling Rate: ',sr,'Hz')
#     print('Duration: ',len(y)/sr)
#     print('Number of samples: ', len(y))
    
#     plt.figure()
#     plt.subplot(3,1,1)
#     librosa.display.waveplot(y, sr)
    
#     plt.subplot(3,1,2)
#     melspec = librosa.feature.melspectrogram(y, sr, n_fft=2048, hop_length=512)
#     logmelspec = librosa.power_to_db(melspec)   
#     librosa.display.specshow(logmelspec, sr=sr)
    
#     plt.subplot(3,1,3)
#     mfcc = librosa.feature.mfcc(y, sr, dct_type=2, n_mfcc=40)
#     librosa.display.specshow(mfcc, sr=sr, x_axis='time', y_axis='mel')    

# y, sr = librosa.load('./sound_classification_50/test/0.wav')
# wave_show(y, sr)   

# y, sr = librosa.load('./sound_classification_50/test/1.wav')
# wave_show(y, sr)

# y, sr = librosa.load('./sound_classification_50/test/2.wav')
# wave_show(y, sr)


### get wave features
# def mfcc_extractor(file_name):
#     y, sr = librosa.load(file_name)
#     mfcc = librosa.feature.mfcc(y, sr, dct_type=2, n_mfcc=40)
#     return mfcc

# def mel_extractor(file_name):
#     y, sr = librosa.load(file_name)
#     melspec = librosa.feature.melspectrogram(y, sr, n_fft=2048, hop_length=512)
#     return librosa.power_to_db(melspec)

# import os

# path = './sound_classification_50/train/'
# dirs = os.listdir(path)

# mfcc_list = []
# mel_list = []
# label_list = []

# for file_name in dirs:
#     mel = mel_extractor(path+file_name)
#     mfcc = mfcc_extractor(path+file_name)   
#     label = file_name.split('.')[0].split('-')[-1]
    
#     mfcc_list.append(mfcc)
#     mel_list.append(mel)
#     label_list.append(float(label))

# np.save('./data/train_data_mfcc.npy', np.array(mfcc_list))
# np.save('./data/train_data_mel.npy', np.array(mel_list))
# np.save('./data/label.npy', np.array(label_list))
    

### convert features to jpg
def convert_to_jpg(features, out_path, dpi=300):
    n = features.shape[0]
    for i in range(n):
        mel = features[i,:,:]
        plt.figure(figsize=[0.5,0.5])
        ax = plt.gca()
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_frame_on(False)
        
        librosa.display.specshow(mel)
        filename  = out_path + str(i) + '.jpg'
        plt.savefig(filename, dpi=dpi, bbox_inches='tight',pad_inches=0)
        plt.cla()
        plt.clf()
        plt.close('all')

mel_data = np.load('./data/train_data_mel.npy')
mfcc_data = np.load('./data/train_data_mfcc.npy')

jpg_path = './data/jpg_data/mel_data2/'
convert_to_jpg(mel_data, jpg_path, dpi=600)

