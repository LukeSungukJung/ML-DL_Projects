import tensorflow as  tf
import numpy as np
from tensorflow.contrib import learn
from sklearn.datasets import load_boston
from sklearn import datasets,metrics,preprocessing
boston = load_boston()


x_data = preprocessing.StandardScaler().fit_transform(boston.data)

"""
with tf.name_scope('inference') as scope:
    w = tf.Variable(tf.zeros([1,13],dtype=tf.float64,name="weight"))
    b = tf.Variable(0,dtype=tf.float64,name="bias")
    y_pred = tf.matmul(w,tf.transpose(x))+b
    
with tf.name_scope('loss') as scope:
    cost = tf.reduce_mean(tf.square(y_true-y_pred))
    
with tf.name_scope('train') as scope:
    optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(cost)
    
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(200):
        sess.run(optimizer, feed_dict = {x:x_data,y_true:y_data})
        mse = sess.run(cost,{x:x_data,y_true:y_data})
        
        print(mse)
        
        
        
        abstraction begin! ㅎㅎ
        ㅁ
        ㅁ
        ㅁ
        ㅁ
        ㅁ
        ㅁ
        ㅁ
        V
"""
NUM_STEP = 200
MINI_BATCH = 506
feature_columns = learn.infer_real_valued_columns_from_input(x_data)
reg = learn.LinearRegressor(feature_columns=feature_columns,
                            optimizer= tf.train.GradientDescentOptimizer(learning_rate=0.1))

reg.fit(x_data,boston.target,steps=NUM_STEP,batch_size=MINI_BATCH)
MSE = reg.evaluate(x_data,boston.target,steps=1)
print(MSE)
