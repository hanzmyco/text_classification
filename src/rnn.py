import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
sys.path.append('..')
import tensorflow as tf
import models
import config

class RNN(models.BaseModel):
    def __init__(self,model):
        models.BaseModel.__init__(self,model)
        self.hidden_sizes = config.HIDDEN_SIZE
        #self.attention_size = config.ATTENTION_SIZE
        self.num_topics = config.NUM_TOPICS
        self.dropout_keep_prob = config.DROPOUT_KEEP_PROB


    def create_actual_model(self, embd):
        return tf.nn.dropout(embd,self.dropout_keep_prob)


    def get_hidden_states(self):

        pass

    def get_logits(self):
        if not config.SELF_ATTENTION_TAG:
            if config.MODEL_NAME !='LSTM':
                if config.DROPOUT_KEEP_PROB!=1.0:
                    with tf.name_scope('dropouts'):
                        last_state_drop = tf.nn.dropout(self.out_state[len(self.hidden_sizes) - 1],self.dropout_keep_prob)
                    self.logits = tf.layers.dense(last_state_drop, self.num_classes, None)
                else:
                    self.logits = tf.layers.dense(self.out_state[len(self.hidden_sizes) - 1], self.num_classes, None)

            else:
                if config.DROPOUT_KEEP_PROB!=1.0:
                    with tf.name_scope('dropouts'):
                        last_state_drop = tf.nn.dropout(self.out_state[len(self.hidden_sizes) - 1][1],self.dropout_keep_prob)
                    self.logits = tf.layers.dense(last_state_drop, self.num_classes, None)

                else:
                    self.logits = tf.layers.dense(self.out_state[len(self.hidden_sizes) - 1][1], self.num_classes, None)
        else:
            #if config.MODEL_NAME !='LSTM':
            self.self_attention()
            self.logits = tf.layers.dense(self.attention_logits, self.num_classes, None)
            print('test attention')
            #else:
                #pass

    def self_attention(self,self_attention_tag = config.SELF_ATTENTION_TAG):

        if self_attention_tag:
            W_s1 = tf.get_variable('attention_matrix',
                                           shape=[config.ATTENTION_SIZE, 2*self.hidden_sizes[-1]],
                                           initializer=tf.random_uniform_initializer())
            w_s2 = tf.get_variable('attention_vector',
                                           shape=[self.num_topics, config.ATTENTION_SIZE],
                                           initializer=tf.random_uniform_initializer())

            self.A = tf.nn.softmax(tf.map_fn(
                lambda x:tf.matmul(w_s2,x),
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
            tile_eye = tf.tile(tf.eye(self.num_topics), [tf.shape(self.A)[0], 1])
            tile_eye = tf.reshape(tile_eye, [-1, self.num_topics, self.num_topics])
            AA_T = tf.matmul(self.A, A_T) - tile_eye
            self.loss += config.ATTENTION_COEF*tf.square(tf.norm(AA_T, axis=[-2, -1], ord='fro'))

        else:
            pass
