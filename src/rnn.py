""" A clean, no_frills character-level generative language model.

CS 20: "TensorFlow for Deep Learning Research"
cs20.stanford.edu
Danijar Hafner (mail@danijar.com)
& Chip Huyen (chiphuyen@cs.stanford.edu)
Lecture 11
"""
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys

sys.path.append('..')
import time

import tensorflow as tf

import utils

#import word2vec_utils
import models

class CharRNN(models.BaseModel):
    def create_actual_model(self, seq):
        layers = [tf.nn.rnn_cell.GRUCell(size) for size in self.hidden_sizes]
        cells = tf.nn.rnn_cell.MultiRNNCell(layers)
        batch = tf.shape(seq)[0]
        zero_states = cells.zero_state(batch, dtype=tf.float32)
        self.in_state = tuple([tf.placeholder_with_default(state, [None, state.shape[1]])
                               for state in zero_states])
        # this line to calculate the real length of seq
        # all seq are padded to be of the same length, which is num_steps
        length = tf.reduce_sum(tf.reduce_max(tf.sign(seq), 2), 1)
        self.output, self.out_state = tf.nn.dynamic_rnn(cells, seq, length, self.in_state)





def main():
    model = 'trump_tweets_short'
    utils.safe_mkdir('checkpoints')
    utils.safe_mkdir('checkpoints/' + model)

    lm = CharRNN(model)
    lm.create_model()
    lm.train()


if __name__ == '__main__':
    main()