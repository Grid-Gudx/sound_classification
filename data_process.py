# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 14:33:12 2021

@author: gdx
"""

import pandas as pd
import numpy as np
import seaborn as sns
# import matplotlib.pyplot as plt

mel_data = np.load('./data/train_data_mel.npy')
label = np.load('./data/label.npy')

### 确认标签列的各类别数量
def class_ratio(label, distribution=True):
    """
    Parameters
    ----------
    label : np.array([1,2,3,..]) 1D array
        DESCRIPTION.
    distribution : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    None.

    """
    cl = np.unique(label)
    n = label.size #sample num
    for i in cl:
        m = np.sum(label==i)
        print(i, m, m/n)
    print('num ',n)
    if distribution:
        sns.distplot(label)

class_ratio(label)


### 分离验证集
from sklearn.model_selection import StratifiedKFold

x = np.array(range(1600)) #jpg name
y = label.astype(np.float32)

out_path = './data/jpg_data/mel_idx'

skf = StratifiedKFold(n_splits=5, random_state=None, shuffle=False) #分层交叉验证
i = 0
for train_index, test_index in skf.split(x, y):
    i += 1
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    np.save(out_path+'/train_data_'+str(i)+'.npy',x_train)
    np.save(out_path+'/val_data_'+str(i)+'.npy',x_test)   
    np.save(out_path+'/train_label_'+str(i)+'.npy',y_train)
    np.save(out_path+'/val_label_'+str(i)+'.npy',y_test)
    
    


