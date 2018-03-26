import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
sys.path.append('..')
import tensorflow as tf
import utils
import models



class CNN(models.BaseModel):
    def __init__(self,model):
        self.base_init(model)
        self.hidden_sizes = [128, 256]

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

    def get_logits(self):
        self.logits = tf.layers.dense(self.out_state[len(self.hidden_sizes) - 1], self.num_classes, None)





def main():
    model = 'trump_tweets_short'
    utils.safe_mkdir('checkpoints')
    utils.safe_mkdir('checkpoints/' + model)

    lm = CNN(model)
    lm.create_model()
    lm.train()


if __name__ == '__main__':
    main()