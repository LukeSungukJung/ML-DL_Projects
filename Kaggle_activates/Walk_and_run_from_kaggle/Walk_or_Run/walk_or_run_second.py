import os
import numpy as np
import tensorflow as tf
import matplotlib.pylab as plt
import pandas as pd
import glob
from IPython.display import Image
import tflearn     
import cv2
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, avg_pool_2d
from tflearn.layers.estimator import regression



PATH =  ""
train_run = os.path.join(PATH,'train','run')
train_run = glob.glob(os.path.join(train_run,"*.png"))

train_walk = os.path.join(PATH,'train','walk')
train_walk = glob.glob(os.path.join(train_walk,"*.png"))

train = pd.DataFrame()
train['file'] = train_run + train_walk
train.head()
train = train['file'].values.tolist()

train_img = [cv2.imread(data)for data in train]
train_img = np.asarray(train_img,dtype=np.int64)
train_label = [1]*len(train_run)+[0]*len(train_walk)
train_label = np.reshape(train_label,[-1,1])

##train data

test_run = os.path.join(PATH,'test','run')
test_run = glob.glob(os.path.join(test_run,'*.png'))
test_walk = os.path.join(PATH,'test','walk')
test_walk = glob.glob(os.path.join(test_walk,'*.png'))

test_label =[1]*len(test_run) + [0]*len(test_walk)
test_label = np.reshape(test_label,[-1,1])
test = pd.DataFrame()
test['label'] = test_run + test_walk
test.head()
test =test['label'].values.tolist()

test_img = [cv2.imread(data) for data in test]
test_img = np.asarray(test_img,dtype=np.int64)

CNN = input_data(shape=[None, 224, 224, 3],name="input_x")


CNN = conv_2d(CNN,32,7,activation='relu', regularizer="L2")
CNN = avg_pool_2d(CNN,2)
CNN = dropout(CNN,keep_prob=0.5)

CNN = conv_2d(CNN,45,5,activation='relu', regularizer="L2")
CNN = avg_pool_2d(CNN,2)
CNN = dropout(CNN,keep_prob=0.5)

CNN = conv_2d(CNN,10,2,activation='relu',regularizer='L2')
CNN = avg_pool_2d(CNN,2)

fl = fully_connected(CNN,1,activation='softmax')
output  = regression(fl,learning_rate=0.0005,loss='binary_crossentropy',name='targets')

model = tflearn.DNN(output,tensorboard_verbose=0,tensorboard_dir = './walk_run',checkpoint_path = './walk_run/checkpoint')
model.fit({'input_x':train_img},{'targets':train_label},show_metric=True,n_epoch=20,batch_size=600)
model.evaluate({'input_x':test_img},{'targets':test_label})



 