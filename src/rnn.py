import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
sys.path.append('..')
import tensorflow as tf
import models
import config

class RNN(models.BaseModel):
    def __init__(self,model):
        self.base_init(model)
        self.hidden_sizes = [128, 256]

    def create_actual_model(self, embd):
        with tf.name_scope("rnn_cell"):
            layers = [tf.nn.rnn_cell.GRUCell(size) for size in self.hidden_sizes]
            cells = tf.nn.rnn_cell.MultiRNNCell(layers)
            batch = tf.shape(embd)[0]
            zero_states = cells.zero_state(batch, dtype=tf.float32)
            self.in_state = tuple([tf.placeholder_with_default(state, [None, state.shape[1]])
                                   for state in zero_states])

            # this line to calculate the real length of seq
            # all seq are padded to be of the same length, which is num_steps

            length = tf.reduce_sum(tf.reduce_max(tf.sign(embd), 2), 1)

            self.output, self.out_state = tf.nn.dynamic_rnn(cells, embd, length, self.in_state)

    def get_logits(self):
        self.logits = tf.layers.dense(self.out_state[len(self.hidden_sizes) - 1], self.num_classes, None)
