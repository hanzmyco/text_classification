import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
sys.path.append('..')
import tensorflow as tf

W_s1 = tf.get_variable('attention_matrix',shape=[512,256, 10],initializer=tf.random_uniform_initializer())

W_s2 = tf.get_variable('attention_matrix2',shape=[1,15, 256],initializer=tf.random_uniform_initializer())

intermediate = tf.tensordot(W_s2,W_s1,axes=[[2],[1]])
b=tf.unstack(intermediate,axis=0)[0]
a=tf.transpose(b,perm=[1,0,2])

print('stop')

