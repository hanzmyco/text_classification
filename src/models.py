import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
sys.path.append('..')
import time
import tensorflow as tf
import utils
import word2vec_utils
import config


class BaseModel(object):
    def base_init(self,model):
        self.model = model
        self.path = '../data/' + model + '.txt'
        self.temp = tf.constant(1.5)
        self.batch_size = config.BATCH_SIZE
        self.lr = config.LR
        self.skip_step = 1
        self.len_generated = 200
        self.gstep = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        self.num_classes = config.NUM_CLASSES
        self.out_state = None
        self.in_state = None
        self.sample = None
        self.num_steps = config.NUM_STEPS  # for RNN unrolled, actually use it for cut down
        self.embedding_size = config.EMBEDDING_SIZE
        self.vocab_size=config.VOCAB_SIZE
        self.loss=0

    def __init__(self, model):
        pass


    def create_actual_model(self, embd):
        pass

    def get_logits(self):
        pass

    def create_model(self,one_hot=False,training=True):

        if one_hot:  # not using embedding layer
            embed = self.seq

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
                embed = tf.nn.embedding_lookup(embed_matrix, self.seq, name='embedding')

        self.create_actual_model(embed)

        self.get_logits()

        _, self.acc_op = tf.metrics.accuracy(labels=tf.argmax(input=self.label, axis=2), predictions=tf.argmax(input=self.logits, axis=1),name = 'my_metrics')

        if training:
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,
                                                           labels=self.label)
            self.loss += loss

            self.loss = tf.reduce_sum(self.loss)

            params = tf.trainable_variables()

            optimizer = tf.train.AdamOptimizer(self.lr)

            grad_and_vars = tf.gradients(self.loss,params)

            clipped_gradients , _= tf.clip_by_global_norm(grad_and_vars,0.5)

            self.opt = optimizer.apply_gradients(zip(clipped_gradients,params),global_step = self.gstep)


            #self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss, global_step=self.gstep)



    def train_one_epoch(self,sess,saver,init,writer,epoch,iteration):
        start_time = time.time()
        sess.run(init)
        total_loss=0
        n_batches = 0
        checkpoint_name = config.CPT_PATH + '/'
        total_accuracy =0
        try:
            while True:
                batch_loss, _ ,accuracy= sess.run([self.loss, self.opt,self.acc_op])

                iteration += 1
                total_loss +=batch_loss
                total_accuracy+=accuracy
                n_batches +=1

        except tf.errors.OutOfRangeError:
            pass

        saver.save(sess, checkpoint_name, iteration)
        print('Average loss and accuracy at epoch {0}: {1},{2}'.format(epoch, total_loss / n_batches,total_accuracy/n_batches))
        print('Took: {0} seconds'.format(time.time() - start_time))
        return iteration

    def train(self,n_epochs):
        writer = tf.summary.FileWriter('../graphs/gist', tf.get_default_graph())

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # initilize accuracy
            running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='my_metrics')
            running_vars_initializer = tf.variables_initializer(var_list=running_vars)
            sess.run(running_vars_initializer)

            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(config.CPT_PATH+ '/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            iteration = self.gstep.eval()
            for epoch in range(n_epochs):
                iteration = self.train_one_epoch(sess, saver, self.init, writer, epoch, iteration)

        writer.close()

    # deprecated
    def train_old(self):
        saver = tf.train.Saver()
        start = time.time()
        min_loss = None

        with tf.Session() as sess:
            writer = tf.summary.FileWriter('../graphs/gist', sess.graph)
            sess.run(tf.global_variables_initializer())
            sess.run(self.init)

            ckpt = tf.train.get_checkpoint_state(os.path.dirname(config.CPT_PATH+ '/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            iteration = self.gstep.eval()

            while True:
                batch_loss, _ = sess.run([self.loss, self.opt])
                if (iteration + 1) % self.skip_step == 0:
                    print('Iter {}. \n    Loss {}. Time {}'.format(iteration, batch_loss, time.time() - start))
                    start = time.time()
                    checkpoint_name = config.CPT_PATH+'/'
                    if min_loss is None:
                        saver.save(sess, checkpoint_name, iteration)
                    elif batch_loss < min_loss:
                        saver.save(sess, checkpoint_name, iteration)
                        min_loss = batch_loss
                iteration += 1

    def _check_restore_parameters(self,sess, saver):
        """ Restore the previously trained parameters if there are any. """
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(config.CPT_PATH + '/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            print("Loading parameters for text classifier")
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("Initializing fresh parameters for text classifier")

    def inference(self):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(self.init)

            if hasattr(config, 'INFERENCE_LABEL_NAME'):
                # initilize accuracy
                running_vars=tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES,scope = 'my_metrics')
                running_vars_initializer = tf.variables_initializer(var_list=running_vars)
                sess.run(running_vars_initializer)

            self._check_restore_parameters(sess, saver)
            output_file = open(config.PROCESSED_PATH+config.INFERENCE_RESULT_NAME,'a+')

            try:
                while True:
                    if hasattr(config, 'INFERENCE_LABEL_NAME'):
                        probability,classes,acc = sess.run([tf.nn.softmax(self.logits, name='softmax_tensor'), tf.argmax(input=self.logits, axis=1),self.acc_op])
                        print(acc)

                    else:
                        probability, classes = sess.run(
                            [tf.nn.softmax(self.logits, name='softmax_tensor'), tf.argmax(input=self.logits, axis=1)])

                    #print(probability)
                    for ite in classes:
                        output_file.write(str(ite)+'\n')

            except tf.errors.OutOfRangeError:
                output_file.close()
                pass



