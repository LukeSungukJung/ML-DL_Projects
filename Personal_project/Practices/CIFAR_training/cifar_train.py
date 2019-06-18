import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets.cifar10 import load_data
 
 
(x_train, y_train), (x_test, y_test) = load_data()
 
y_tf_train = tf.squeeze(tf.one_hot(y_train,10),axis=1)
 
 
class Cnn_conv:
    filter_arr= [] 
    last_layer =None
    x_data = None
    initializer = None
    stride_arr =[]
    kernal_arr =[]
    last_size = 1
    def make_filter_layers(self,num):
        pre_channel= 3
        
        for i in range(num):
            stride = int(input(str(i+1)+'th how many you want to make stides?'))
            self.stride_arr.append([1,stride,stride,1])
            kernal_n = int(input(str(i+1)+'th how many you want to make kernals?'))
            self.kernal_arr.append([1,kernal_n,kernal_n,1])
            if i==0:
                filter_ = tf.Variable(self.initializer([4,4,pre_channel,4]))
                pre_channel = 4
                self.filter_arr.append(filter_)
            else:
                channel =  int(input(str(i+1)+'th how many you want to make channels?'))
                filter_ = tf.Variable(self.initializer([kernal_n,kernal_n,pre_channel,channel]))
                self.filter_arr.append(filter_)
                pre_channel = channel                
                
        
    def __init__(self,input_x,num_of_layers=2,keep_prob = 0.5):
        self.initializer = tf.contrib.layers.xavier_initializer()
        self.make_filter_layers(num_of_layers)
        std_stride = [1,1,1,1]
        for i in range(num_of_layers):
            if i==0:
                convol_unit = tf.nn.conv2d(input_x,filter=self.filter_arr[i],
                                           strides=std_stride,padding='SAME')
            else:
                convol_unit = tf.nn.conv2d(self.last_layer,filter=self.filter_arr[i],
                                           strides=std_stride,padding='SAME')
                
                
            convol_unit = tf.nn.relu(convol_unit)
            convol_unit = tf.nn.max_pool(convol_unit,
                                         ksize= self.kernal_arr[i],
                                         strides=self.stride_arr[i],
                                         padding='SAME')
            convol_unit = tf.nn.dropout(convol_unit, keep_prob=keep_prob)
            
            self.last_layer = convol_unit
        last_data_width = self.last_layer.shape[-3]
        last_data_height= last_data_width
        last_channel =self.last_layer.shape[-1]
        self.last_size= last_channel*last_data_height*last_data_width   
        
    def return_conv_output(self):
        return self.last_layer,int(self.last_size)
    
    
class fully_connected_layers:
    conv_weight_arr = []
    conv_bias_arr = []
    last_output = None
    input_size = None
    input_data= None
    
    def __init__(self,num_of_layers,input_x,data_size):
        self.input_data = input_x
        initializer = tf.contrib.layers.xavier_initializer()
        pre_size = data_size
        self.input_size = data_size
        for i in range(num_of_layers):
            if i==0:
                self.conv_weight_arr.append(tf.Variable(initializer([pre_size,250])))
                pre_size = 250
                self.conv_bias_arr.append(tf.Variable(tf.zeros([pre_size])))
            elif i == num_of_layers-1:
                self.conv_weight_arr.append(tf.Variable(initializer([pre_size,10])))
                self.conv_bias_arr.append(tf.Variable(tf.zeros([10])))
            else:
                self.conv_weight_arr.append(tf.Variable(initializer([pre_size,100])))
                pre_size = 100
                self.conv_bias_arr.append(tf.Variable(tf.zeros([pre_size])))
                
    def forward(self,keep_prob=0.5):
        dept_layers = len(self.conv_bias_arr)
        pre_output = tf.reshape(self.input_data,shape=[-1,self.input_size])
        for i in range(dept_layers):
                affine_process= tf.nn.relu(
                        tf.matmul(pre_output,self.conv_weight_arr[i])+self.conv_bias_arr[i])
                affine_process = tf.nn.dropout(affine_process, keep_prob=keep_prob)
                pre_output = affine_process
        self.last_output = pre_output
        return self.last_output
    
    def cost_func(self,Y):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.last_output,labels=Y))
    
    def accuracy_func(self,Y):
        acc =  tf.reduce_mean(tf.cast(tf.equal(self.last_output,Y),dtype=tf.float32))
        return acc
    
X = tf.placeholder(dtype=tf.float32,shape=[None,32,32,3])
 
Y = tf.placeholder(dtype=tf.float32,shape=[None,10])
keep_prob = tf.placeholder(tf.float32)
conv_num = int(input('the number of convoulutional layers :'))
full_num = int(input('the number of fully connected layers :'))
cnn =  Cnn_conv(X,conv_num)
conv_output,size = cnn.return_conv_output()
fl_l = fully_connected_layers(full_num,conv_output,size)
 
hypothesis = fl_l.forward()
cost = fl_l.cost_func(Y)
accuracy = fl_l.accuracy_func(Y)
optimizer = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(cost)
 
 
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    t1,t2 =sess.run([X,Y],feed_dict = {X:x_train,Y:y_tf_train.eval(),keep_prob:0.5})
 
    for i in range(1001):
        optimizer_ = sess.run([optimizer],feed_dict = {
                X:x_train,Y:y_tf_train.eval(),keep_prob:0.5})
        cost_,acc_ = sess.run([cost,accuracy],feed_dict = {
                X:x_train,Y:y_tf_train.eval(),keep_prob:1})
        if i %3==0:
            print(i+1,'th acc:',acc_,', cost:',cost_)