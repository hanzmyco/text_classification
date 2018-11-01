import rnn
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
sys.path.append('../../')
import tensorflow as tf
import config

class LSTM(rnn.RNN):
    def create_actual_model(self, embd):
        embd=rnn.RNN.create_actual_model(self,embd)
        
        with tf.name_scope("rnn_cell"):
            layers = [tf.nn.rnn_cell.LSTMCell(size,state_is_tuple=True,initializer=self.initializer) for size in self.hidden_sizes]
            cells = tf.nn.rnn_cell.MultiRNNCell(layers,state_is_tuple=True)
            batch = tf.shape(embd)[0]
            zero_tuples = cells.zero_state(batch, dtype=tf.float32)

            # zero state are tuples, need to be handled specifically
            in_state = tuple(
                    [tf.nn.rnn_cell.LSTMStateTuple(tf.unstack(state_tuple,axis =0)[0], tf.unstack(state_tuple,axis =0)[1]) for state_tuple in zero_tuples])

            length = tf.cast(tf.reduce_sum(tf.reduce_max(tf.sign(embd), 2), 1),tf.int32)


            if config.MODEL_BI_DIRECTION:
                bw_layers = [tf.nn.rnn_cell.LSTMCell(size,state_is_tuple=True) for size in self.hidden_sizes]
                bw_cells = tf.nn.rnn_cell.MultiRNNCell(bw_layers)
                bw_zero_tuples = bw_cells.zero_state(batch,dtype=tf.float32)
                bw_in_state = tuple(
                    [tf.nn.rnn_cell.LSTMStateTuple(tf.unstack(state_tuple, axis=0)[0],
                                                   tf.unstack(state_tuple, axis=0)[1]) for state_tuple in bw_zero_tuples])

                self.output, self.out_state = tf.nn.bidirectional_dynamic_rnn(cells, bw_cells, embd, length, in_state,
                                                                              bw_in_state)
                self.output = tf.concat([self.output[0], self.output[1]], 2)

                out_c_0 = tf.concat([self.out_state[0][0][0],self.out_state[1][0][0]],1)
                out_h_0 = tf.concat([self.out_state[0][0][1],self.out_state[1][0][1]],1)
                out_c_1 = tf.concat([self.out_state[0][1][0],self.out_state[1][1][0]],1)
                out_h_1 = tf.concat([self.out_state[0][1][1], self.out_state[1][1][1]], 1)

                self.out_state=tuple((tf.nn.rnn_cell.LSTMStateTuple(out_c_0,out_h_0),tf.nn.rnn_cell.LSTMStateTuple(out_c_1,out_h_1)))

                #self.out_state = tuple((tf.concat([self.out_state[0][0], self.out_state[1][0]], 1),
                #                        tf.concat([self.out_state[0][1], self.out_state[1][1]], 1)))
            else:
                self.output, self.out_state = tf.nn.dynamic_rnn(cells, embd, length, in_state)
