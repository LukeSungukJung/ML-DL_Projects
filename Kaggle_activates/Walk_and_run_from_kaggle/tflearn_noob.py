import tensorflow as tf
import numpy as np
import tflearn
import matplotlib.pyplot as plt
from tflearn.layers.core import input_data as core_input_data, dropout, fully_connected
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
x_train =mnist.train.images
y_train =mnist.train.labels
x_test = mnist.test.images
y_test = mnist.test.labels

X = core_input_data(shape=[None,784],name='input_x')
Y = tf.placeholder(tf.float64,shape=([None,10]),name="labels")

fc_layer = fully_connected(X,n_units=128,activation='relu')
fc_layer = fully_connected(fc_layer,n_units=10,activation='softmax')
output = regression(fc_layer,optimizer='adam',learning_rate=0.001,loss='categorical_crossentropy',name='targets')

model = tflearn.models.DNN(output)
model.fit({'input_x':x_train},{'targets':y_train})
model.evaluate({'input_x':x_test},{'targets':y_test})