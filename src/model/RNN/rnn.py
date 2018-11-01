import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../Attention/')

import tensorflow as tf
import models
import config
import SelfAttention

class RNN(models.BaseModel):
    def __init__(self,model):
        models.BaseModel.__init__(self,model)


    def create_actual_model(self, embd):
        return tf.nn.dropout(embd,config.DROPOUT_KEEP_PROB)


    def get_hidden_states(self):
        pass

    def lstm_logits(self):
        if config.DROPOUT_KEEP_PROB!=1.0:
            with tf.name_scope('dropouts'):
                last_state_drop = tf.nn.dropout(self.out_state[len(config.HIDDEN_SIZE) - 1][1],config.DROPOUT_KEEP_PROB)
            self.logits = tf.layers.dense(last_state_drop, config.NUM_CLASSES, None)

        else:
            self.logits = tf.layers.dense(self.out_state[len(config.HIDDEN_SIZE) - 1][1], config.NUM_CLASSES, None)

    def gru_logits(self):
        if config.DROPOUT_KEEP_PROB!=1.0:
            with tf.name_scope('dropouts'):
                last_state_drop = tf.nn.dropout(self.out_state[len(config.HIDDEN_SIZE) - 1],config.DROPOUT_KEEP_PROB)
            self.logits = tf.layers.dense(last_state_drop, config.NUM_CLASSES, None)
        else:
            self.logits = tf.layers.dense(self.out_state[len(config.HIDDEN_SIZE) - 1], config.NUM_CLASSES, None)


    def get_logits(self):
        if not config.ATTENTION_TAG:
            if config.MODEL_NAME !='LSTM':
                self.gru_logits()
            else:
                self.lstm_logits()
        else:
            if config.SELF_ATTENTION_TAG:
                sf_instance = SelfAttention.SelfAttention()
                self.A,self.attention_logits=sf_instance.createAttention(self.output,self.loss)
            elif config.CNN_ATTENTION_TAG:
                pass
            self.logits = tf.layers.dense(self.attention_logits, config.NUM_CLASSES, None)


    def self_Attention(self):
        print('self-attention')
        W_s1 = tf.get_variable('attention_matrix',
                                       shape=[config.ATTENTION_SIZE, 2*config.HIDDEN_SIZE[-1]],
                                       initializer=tf.random_uniform_initializer())
        W_s2 = tf.get_variable('attention_vector',
                                       shape=[config.NUM_TOPICS, config.ATTENTION_SIZE],
                                       initializer=tf.random_uniform_initializer())

        self.A = tf.nn.softmax(tf.map_fn(
            lambda x:tf.matmul(W_s2,x),
            tf.tanh(
                tf.map_fn(
                    lambda x : tf.matmul(W_s1,tf.transpose(x)),
                self.output)
            )
        ),name='Attention_matrix')

        M = tf.matmul(self.A,self.output)

        # currently support topics with same weight
        self.attention_logits = tf.reduce_sum(M,1)

        A_T = tf.transpose(self.A, perm=[0, 2, 1])
        tile_eye = tf.tile(tf.eye(config.NUM_TOPICS), [tf.shape(self.A)[0], 1])
        tile_eye = tf.reshape(tile_eye, [-1, config.HIDDEN_SIZE, config.HIDDEN_SIZE])
        AA_T = tf.matmul(self.A, A_T) - tile_eye

        # regularlization term
        self.loss += config.ATTENTION_COEF*tf.square(tf.norm(AA_T, axis=[-2, -1], ord='fro'))
