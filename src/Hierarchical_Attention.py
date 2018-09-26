import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
sys.path.append('..')
import tensorflow as tf
import models
import config
import rnn

class HAN(rnn.RNN):
    def __init__(self,model):
        rnn.RNN.__init__(self,model)

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
            self.self_attention()
            self.logits = tf.layers.dense(self.attention_logits, self.num_classes, None)
            print('self-attention')
        
