import numpy as np
import tensorflow as tf
from pandas.io.parsers import read_csv

data = read_csv('Transaction_Bread.csv',sep=',')
xy = np.array(data, dtype = np.float32)

W = tf.Variable(tf.random_normal([2,1],dtype=np.float32),name="weight")
b = tf.Variable(tf.random_normal([1],dtype=np.float32),name="bias")
x_data = xy[:,1:3]
y_data = xy[:,3:4]

##y_std = np.array([(y_data[:,0] - y_data[:,0].mean())/y_data[:,0].std()]).T
##전처리

X = tf.placeholder(tf.float32, shape=[None,2])
Y = tf.placeholder(tf.float32, shape=[None,1])

H = tf.matmul(X,W)+b

cost = tf.reduce_mean(tf.square(H-Y))

optimizer = tf.train.AdamOptimizer(learning_rate=0.0005)
train = optimizer.minimize(cost)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for i in range(50001):
    cost_,hypo_,_ = sess.run([cost,H,train],feed_dict={X:x_data,Y:y_data})
    if i%1000==0:
        print("step:",i,",",cost_,",",hypo_[0])
        

saver = tf.train.Saver()
save_path = saver.save(sess,'./transaction.cpkt')
print("save success!")