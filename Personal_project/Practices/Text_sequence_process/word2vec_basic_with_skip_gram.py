import tensorflow as tf
import numpy as np
import math
batch_size = 64
embedding_dimension = 5
negative_samples = 8

digit_2_map = {1:'one',2:'Two',3:'Three',4:'Four',5:'Five',6:'Six',7:'Seven',8:'Eight',9:'Nine'}

sentences = []

for i in range(10000):
    rand_odd_ints = np.random.choice(range(1,10,2),3)
    rand_even_ints = np.random.choice(range(2,10,2),3)
    sentences.append(" ".join([digit_2_map[r] for r in rand_odd_ints]))
    sentences.append(" ".join([digit_2_map[r] for r in rand_even_ints]))
    
word2index ={}
index = 0
for words in sentences:
    for word in words.lower().split():
        if word not in word2index:
            word2index[word] = index
            index+=1

index2word = {index:word for index,word in word2index.items()}
voca_size = len(index2word)

skip_gram_pairs = []
for sent in sentences:
    tokenized_sent = sent.lower().split()
    for i in range(1,len(tokenized_sent)-1):
        word_context_pair = [[word2index[tokenized_sent[i-1]],
                             word2index[tokenized_sent[i+1]]],word2index[tokenized_sent[i]]]
        skip_gram_pairs.append([word_context_pair[1],word_context_pair[0][0]])
        skip_gram_pairs.append([word_context_pair[1],word_context_pair[0][1]])
        

def get_skip_gram_batch(batch_size):
    instance_indices = list(range(len(skip_gram_pairs)))
    np.random.shuffle(instance_indices)
    batch = instance_indices[:batch_size]
    x = [skip_gram_pairs[i][0] for i in batch]
    y = [[skip_gram_pairs[i][1]] for i in batch]
    return x,y

train_inputs = tf.placeholder(tf.int32,shape=[batch_size])
train_labels = tf.placeholder(tf.float32,shape=[batch_size,1])

embeddings = tf.Variable(tf.random_uniform([voca_size,embedding_dimension],-1,1))

embed = tf.nn.embedding_lookup(embeddings,train_inputs)

nce_weights = tf.Variable(
        tf.truncated_normal([voca_size,embedding_dimension],stddev=1.0/math.sqrt(embedding_dimension)))

nce_biases = tf.Variable(tf.zeros([voca_size]))

cost = tf.reduce_mean(
        tf.nn.nce_loss(weights=nce_weights,biases=nce_biases,inputs=embed,labels=train_labels,num_sampled=negative_samples,num_classes=voca_size)
        )
global_step = tf.Variable(0,trainable=False)
learning_Rate = tf.train.exponential_decay(learning_rate=0.1,global_step=global_step,decay_steps=1000,decay_rate=0.95,staircase=True)
train_step = tf.train.GradientDescentOptimizer(learning_Rate).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(1000):
        x_batch,y_batch = get_skip_gram_batch(batch_size)
        _ =sess.run(train_step,feed_dict={train_inputs:x_batch,train_labels:y_batch})
        if step%100 ==0:
            loss_value = sess.run(cost,feed_dict={train_inputs:x_batch,train_labels:y_batch})
            print("LOSS",step,"th loss:",loss_value)
            


