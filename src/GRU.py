import rnn
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
sys.path.append('..')
import tensorflow as tf
import config

class GRU(rnn.RNN):
    def create_actual_model(self, embd):
        with tf.name_scope("rnn_cell"):
            layers = [tf.nn.rnn_cell.GRUCell(size) for size in self.hidden_sizes]
            cells = tf.nn.rnn_cell.MultiRNNCell(layers)

            batch = tf.shape(embd)[0]

            zero_states = cells.zero_state(batch, dtype=tf.float32)

            in_state = tuple([tf.placeholder_with_default(state, [None, state.shape[1]])
                                   for state in zero_states])

            # this line to calculate the real length of seq
            # all seq are padded to be of the same length, which is num_steps

            length = tf.cast(tf.reduce_sum(tf.reduce_max(tf.sign(embd), 2), 1),tf.int32)



            if config.MODEL_BI_DIRECTION:
                bw_layers = [tf.nn.rnn_cell.GRUCell(size) for size in self.hidden_sizes]
                bw_cells = tf.nn.rnn_cell.MultiRNNCell(bw_layers)
                bw_zero_states =bw_cells.zero_state(batch,dtype=tf.float32)
                bw_in_state = tuple([tf.placeholder_with_default(state, [None, state.shape[1]])
                                   for state in bw_zero_states])

                self.output, self.out_state = tf.nn.bidirectional_dynamic_rnn(cells,bw_cells, embd, length, in_state,bw_in_state)
                print('stop here')
            else:
                self.output, self.out_state = tf.nn.dynamic_rnn(cells, embd, length, in_state)



