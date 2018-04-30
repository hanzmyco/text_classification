import rnn
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
sys.path.append('..')
import tensorflow as tf
import models
import config

class LSTM(rnn.RNN):
    def create_actual_model(self, embd):
        with tf.name_scope("rnn_cell"):
            layers = [tf.nn.rnn_cell.LSTMCell(size,state_is_tuple=True) for size in self.hidden_sizes]
            cells = tf.nn.rnn_cell.MultiRNNCell(layers,state_is_tuple=True)
            batch = tf.shape(embd)[0]
            zero_tuples = cells.zero_state(batch, dtype=tf.float32)

            # zero state are tuples, need to be handled specifically
            self.in_state = tuple(
                    [tf.nn.rnn_cell.LSTMStateTuple(tf.unstack(state_tuple,axis =0)[0], tf.unstack(state_tuple,axis =0)[1]) for state_tuple in zero_tuples])

            length = tf.reduce_sum(tf.reduce_max(tf.sign(embd), 2), 1)

            self.output, self.out_state = tf.nn.dynamic_rnn(cells, embd, length, self.in_state)



