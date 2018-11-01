import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
sys.path.append('../../')
import tensorflow as tf
import config

class SelfAttention(object):
    def __init__(self):
        pass

    def createAttention(self,output_Lastlayer,loss):
        print('self-attention')
        W_s1 = tf.get_variable('attention_matrix',
                                       shape=[config.ATTENTION_SIZE, 2*config.HIDDEN_SIZE[-1]],
                                       initializer=tf.random_uniform_initializer())
        W_s2 = tf.get_variable('attention_vector',
                                       shape=[config.NUM_TOPICS, config.ATTENTION_SIZE],
                                       initializer=tf.random_uniform_initializer())

        A = tf.nn.softmax(tf.map_fn(
            lambda x:tf.matmul(W_s2,x),
            tf.tanh(
                tf.map_fn(
                    lambda x : tf.matmul(W_s1,tf.transpose(x)),
                output_Lastlayer)
            )
        ),name='Attention_matrix')

        M = tf.matmul(A,output_Lastlayer)

        # currently support topics with same weight
        attention_logits = tf.reduce_sum(M,1)

        A_T = tf.transpose(A, perm=[0, 2, 1])
        tile_eye = tf.tile(tf.eye(config.NUM_TOPICS), [tf.shape(A)[0], 1])
        tile_eye = tf.reshape(tile_eye, [-1, config.NUM_TOPICS, config.NUM_TOPICS])
        AA_T = tf.matmul(A, A_T) - tile_eye

        # regularlization term
        loss += config.ATTENTION_COEF*tf.square(tf.norm(AA_T, axis=[-2, -1], ord='fro'))

        return A,attention_logits
