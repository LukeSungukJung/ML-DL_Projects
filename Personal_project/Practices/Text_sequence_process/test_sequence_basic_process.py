import numpy as np

import tensorflow as tf
 
 
 
dic_2_words= {
 
        1:'crux',
 
        2:'sunguk ',
 
        3:'gukjang',
 
        4:'maestro',
 
        5:'magician',
 
        6:'code',
 
        7:'caster',
        
        8:'emperor',
        
        9:'overload'}
 
 
 
dic_2_words[0] = 'pad'
 
 
batch_size = 128
 
element_size = 1
 
embedding_dimension = 64
 
num_classfy = 2
 
hidden_size = 10
 
 
time_step=6
 
seqlens = []
 
labels = []
 
 
 
data = []
 
for i in range(5000):
 
    rand_seq_len = np.random.choice(range(1,5))
 
    pad_len = 6-rand_seq_len
 
    seqlens.append(rand_seq_len)
 
    rand_ints = np.random.choice(range(1,10),rand_seq_len)
 
    rand_ints= np.append(rand_ints,[0]*pad_len)
 
    if rand_seq_len%2 ==0:
 
        labels.append([1])
        data.append(" ".join(dic_2_words[r] for r in rand_ints))
 
    else:
 
        labels.append([0])
        data.append(" ".join(dic_2_words[r] for r in rand_ints))
 
word2index = {}
 
voca_size= len(dic_2_words)
 
w2i_index = 0
 
for sentence in data:
 
    for word in sentence.lower().split():
 
        if word not in word2index:
 
            word2index[word]= w2i_index
 
            w2i_index+=1
 
            
 
index2word = {w:i for i,w in word2index.items()}
 
 
 
def get_batch(batch_size,data_x,data_y,seqlen_,sepe = 0):
 
    seq = seqlen_[sepe:batch_size+sepe]
 
    x = [[word2index[word] for word in data_x[i].lower().split()]
 
         for i in seq]
 
    y = data_y[sepe:sepe+batch_size]
 
    return x,y,seq
 
 
 
def get_embedding(data_x):
 
    embedding = tf.Variable(tf.random_uniform([voca_size,embedding_dimension],-1,1))
 
    embed = tf.nn.embedding_lookup(embedding,data_x)
 
    return embed
 
 
 
def fully_connected_layers(x,weight_size,output_size):
 
    w = tf.Variable(tf.truncated_normal([weight_size,output_size],mean=0,stddev=0.01))
 
    b = tf.Variable(tf.truncated_normal([output_size],mean=0,stddev=0.01))
 
    return (tf.matmul(x,w)+b)
 
 
 
def change_onehot(y):
 
    one_hot = []
 
    one_hot= np.append(one_hot,[[0],[0]]*len(y))
 
    one_hot = np.reshape(one_hot,[-1,2])
 
    one_i = 0
 
    for one in y:
 
        one_hot[one_i][one[0]] = 1
 
        one_i+=1
 
    return one_hot
 
 
_labels = labels
train_x,train_y,train_seq = get_batch(batch_size,data,labels,seqlens)
 
train_y = change_onehot(train_y) 
 
input_x = tf.placeholder(tf.int32,shape=[None,time_step])
 
labels = tf.placeholder(tf.float32,shape=[None,num_classfy])
 
seqlens_ = tf.placeholder(tf.int32,shape=[None])
 
embed_x = get_embedding(input_x)
 
 
 
lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size,forget_bias=1)
 
outputs,state = tf.nn.dynamic_rnn(lstm_cell,embed_x,sequence_length=seqlens_,dtype=tf.float32)
 
 
 
output = fully_connected_layers(state[1],hidden_size,num_classfy)
 
 
 
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=output))
 
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
 
acc = tf.reduce_mean(
 
        tf.cast(
 
        tf.equal(tf.argmax(labels,1),tf.argmax(output,1)),dtype=tf.float32)
 
        )
 
 
 
 
 
with tf.Session() as sess:
 
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        train,cost_ = sess.run([optimizer,cost],feed_dict={input_x:train_x,labels:train_y,seqlens_:train_seq})
    
        if i%100==0:
            accuracy = sess.run([acc],feed_dict={input_x:train_x,labels:train_y,seqlens_:train_seq})
    
            print(accuracy,',',cost_)
    test_x,test_y,test_seq =get_batch(batch_size,data,_labels,seqlens,batch_size)
    test_y =change_onehot(test_y)
    
    for k in range(5):
        cost_,accuracy = sess.run([cost,acc],feed_dict={input_x:test_x,labels:test_y,seqlens_:test_seq})
        print(k,'th cost:',cost_,'accuracy:',accuracy)
        output_ = sess.run(outputs,feed_dict={input_x:test_x,labels:test_y,seqlens_:test_seq})
     
 
   