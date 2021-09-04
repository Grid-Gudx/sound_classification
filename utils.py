# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 14:50:42 2021

# 多分类评价指标及训练过程记录

@author: gdx
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score,\
    f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from tensorflow.keras.callbacks import Callback

class mc_metrics():
    """
    example:
    input_type:
        label_type='label'
        y_score=[1.0,2.,3.,1.]
        y_true= [1,2,3,4]
      or
        label_type='one_hot'
            y_score=[[0.8,0],[0,1]]
            y_true= [[1,0],[1,0]]
    
    for pre,recall,f1, and confusion, the multi_class='one vs rest', so if class balance,recall=acc
    """
    def __init__(self, input_label_type='one_hot'):
        self.label_type = input_label_type
    
    def acc(self, y_score, y_true):
        if self.label_type=='one_hot':
            y_pred = np.argmax(y_score, axis=1)
            y_true = np.argmax(y_true, axis=1)
        else:
            y_pred = y_score.round()
        return accuracy_score(y_true, y_pred)
        
    def pre(self, y_score, y_true):
        if self.label_type=='one_hot':
            y_pred = np.argmax(y_score, axis=1)
            y_true = np.argmax(y_true, axis=1)
        else:
            y_pred = y_score.round()
        return precision_score(y_true, y_pred, average='macro')
    
    def recall(self, y_score, y_true):
        if self.label_type=='one_hot':
            y_pred = np.argmax(y_score, axis=1)
            y_true = np.argmax(y_true, axis=1)
        else:
            y_pred = y_score.round()
        return recall_score(y_true, y_pred, average='macro')
    
    def f1(self, y_score, y_true):
        if self.label_type=='one_hot':
            y_pred = np.argmax(y_score, axis=1)
            y_true = np.argmax(y_true, axis=1)
        else:
            y_pred = y_score.round()
        return f1_score(y_true, y_pred, average='macro')
    
    def auc(self, y_score, y_true): #only input one_hot
        '''y_score(i,j) is the j-th label Probability of the i-th sample
        References
        ----------
        `Hand, D.J., Till, R.J. (2001). A Simple Generalisation of the Area
            Under the ROC Curve for Multiple Class Classification Problems.
            Machine Learning, 45(2), 171-186.
            <http://link.springer.com/article/10.1023/A:1010920819831>`_
        '''
        if self.label_type!='one_hot':
            print('error: input is not one_hot')
            return
        return roc_auc_score(y_score, y_true, average='macro',multi_class='ovo')
    
    def confusion(self, y_score, y_true):
        '''return axis=0(row)--true label, axis=1(column)--pred label'''
        if self.label_type=='one_hot':
            y_pred = np.argmax(y_score, axis=1)
            y_true = np.argmax(y_true, axis=1)
        else:
            y_pred = y_score.round()
        return confusion_matrix(y_true,y_pred)
    
    def confusion_plot(self, y_score, y_true):
        """
        Enter the confusion matrix and draw the confusion matrix diagram
        """
        if self.label_type=='one_hot':
            y_pred = np.argmax(y_score, axis=1)
            y_true = np.argmax(y_true, axis=1)
        else:
            y_pred = y_score.round()
        matrix = confusion_matrix(y_true,y_pred)
        
        num_classes = matrix.shape[0]
        
        # plot confusion matrix
        plt.figure()
        matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis] #out Proportion of each label
        plt.imshow(matrix, cmap=plt.cm.Blues)
    
        labels = ['c'+x for x in [str(i) for i in list(range(num_classes))]]
        plt.xticks(range(num_classes), labels)
        plt.yticks(range(num_classes), labels)

        plt.colorbar()
        plt.xlabel('Pred Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion matrix')
    
        thresh = matrix.max() / 2
        for x in range(num_classes):
            for y in range(num_classes):
                info = float(format((matrix[y, x]),'.3f'))
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.show()
        return

class LossHistory(Callback):
    def __init__(self, model_path, history_path='none'):
        self.history_path = history_path
        self.model_path = model_path
        
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accuracy = []
        self.val_loss = []
        self.val_acc = []
        self.highest = 0. #store the best accuracy
        self.highest_epoch = 0

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accuracy.append(logs.get('categorical_accuracy'))
        self.val_loss.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_categorical_accuracy'))
        
        if self.val_acc[epoch] >= self.highest:
            self.highest = self.val_acc[epoch]
            self.highest_epoch = epoch
            self.model.save(self.model_path)
        
        print('epoch: %d | acc: %0.5f | loss: %.5f | val_acc: %0.5f | val_loss: %.5f'\
              %(epoch, self.accuracy[epoch], self.losses[epoch], \
                self.val_acc[epoch], self.val_loss[epoch]))
        print('best epoch is %3d with val_acc:%.5f' \
              %(self.highest_epoch,self.highest))

    def on_train_end(self, logs={}):
        if self.history_path!='none':
            history_data = np.array([self.accuracy, self.losses, self.val_acc, self.val_loss])
            np.save(self.history_path,history_data)
    
    def loss_plot(self):
        iters = range(len(self.losses))
        fig = plt.figure()
        # acc
        ax = fig.add_subplot(111)
        line1=ax.plot(iters, self.accuracy, 'r', label='train acc')
        # val_acc
        line2=ax.plot(iters, self.val_acc, 'b', label='val acc')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        ax1=ax.twinx()#添加一个兄弟轴，共x轴
        # loss
        line3=ax1.plot(iters, self.losses, 'g', label='train loss')
        # val_loss
        line4=ax1.plot(iters, self.val_loss, 'k', label='val loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        lins = line1+line2+line3+line4
        labs = [l.get_label() for l in lins]
        ax.legend(lins, labs)
        plt.grid(True)
        plt.title('Train and Validation loss & accuracy')
        plt.show()
        
if __name__ == '__main__': 
    ### example
    y_true = np.array([2, 0, 2, 2, 0, 1, 1, 6])
    y_score = np.array([0, 0, 2, 2, 0, 2, 1, 6])
    
    metric = mc_metrics(input_label_type='label')
    acc = metric.acc(y_score, y_true)
    pre = metric.pre(y_score, y_true)
    recall = metric.recall(y_score, y_true)
    f1 = metric.f1(y_score, y_true)
    cm = metric.confusion(y_score, y_true)
    metric.confusion_plot(y_score, y_true)
    print(acc,pre,recall,f1)
    
    
    y_score = label_binarize(y_score, list(np.unique(y_true)))
    y_true = label_binarize(y_true, list(np.unique(y_true)))
    metric = mc_metrics(input_label_type='one_hot')
    acc = metric.acc(y_score, y_true)
    pre = metric.pre(y_score, y_true)
    recall = metric.recall(y_score, y_true)
    f1 = metric.f1(y_score, y_true)
    cm = metric.confusion(y_score, y_true)
    metric.confusion_plot(y_score, y_true)
    print(acc,pre,recall,f1)
