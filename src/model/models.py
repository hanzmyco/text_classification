import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
sys.path.append('../')
import tensorflow as tf
import config


class BaseModel(object):
    def __init__(self, model):
        self.model = model
        self.path = '../data/' + model + '.txt'
        self.temp = tf.constant(1.5)
        #config.BATCH_SIZE = config.BATCH_SIZE
        #config.LR = config.LR
        self.skip_step = 1
        self.len_generated = 200
        self.gstep = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        self.out_state = None
        self.in_state = None
        self.sample = None
        config.NUM_STEPS = config.NUM_STEPS  # for RNN unrolled, actually use it for cut down
        self.embedding_size = config.EMBEDDING_SIZE
        self.vocab_size=config.VOCAB_SIZE
        self.loss=0
        self.initializer=None
        print('test git')
        if config.INITIALIZER=='xavier':
            self.initializer=tf.contrib.layers.xavier_initializer()


    def create_actual_model(self, embd):
        pass

    def get_logits(self):
        pass

    def create_model(self,in_data,one_hot=False,training=True):
        with tf.variable_scope('scope_model',reuse=tf.AUTO_REUSE):
            if one_hot:  # not using embedding layer
                embed = in_data[0]

            else:  # using embedding layer

                with tf.name_scope('embed'):
                    if not config.PRETRAIN_EMBD_TAG:

                            embed_matrix = tf.get_variable('embed_matrix',
                                                           shape=[self.vocab_size, self.embedding_size],
                                                           initializer=tf.random_uniform_initializer())


                    else:
                        embed_matrix = tf.Variable(self.pretrain_embd,
                                                   trainable=config.PRETRAIN_EMBD_TRAINABLE,name='embed_matrix')
                        '''
                        #make sure the pretrain embd is load correctly
                        with tf.Session() as sess:
                            sess.run(tf.initialize_all_variables())
                            print(sess.run(embed_matrix))
                        '''
                    embed = tf.nn.embedding_lookup(embed_matrix, in_data[0], name='embedding')
                    print(embed_matrix.name)
                    print(embed.name)

            self.create_actual_model(embed)

            self.get_logits()

            self.logits = tf.layers.batch_normalization(self.logits,name='Batch_Normalization')

            self.logits = tf.nn.relu(self.logits,name='RELU')

            labels = tf.argmax(input=in_data[1], axis=2)
            predictions = tf.argmax(input=self.logits, axis=1)

            _, self.acc_op = tf.metrics.accuracy(labels=labels, predictions=predictions,name = 'my_metrics')


            if training:
                loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,
                                                               labels=in_data[1])
                self.loss += loss

                self.loss = tf.reduce_sum(self.loss)

                params = tf.trainable_variables()

                optimizer = tf.train.AdamOptimizer(config.LR)

                grad_and_vars = tf.gradients(self.loss,params)

                clipped_gradients , _= tf.clip_by_global_norm(grad_and_vars,0.5)

                self.opt = optimizer.apply_gradients(zip(clipped_gradients,params),global_step = self.gstep)

                #self.opt = tf.train.AdamOptimizer(config.LR).minimize(self.loss, global_step=self.gstep)


            print('test git')
