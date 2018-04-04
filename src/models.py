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
        self.seq = tf.placeholder(tf.int32, [None, None])
        self.label = tf.placeholder(tf.int32, [None, None])
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

    def __init__(self, model):
        pass


    def create_actual_model(self, embd):
        pass

    def get_logits(self):
        pass

    def create_model(self,one_hot=False):
        '''
        local_dest = config.DATA_PATH
        words, self.vocab_size, actual_text = word2vec_utils.read_data(local_dest)
        self.vocab, _ = word2vec_utils.build_vocab(words, self.vocab_size, '../visualization')
        self.index_words = word2vec_utils.convert_words_to_index(actual_text, self.vocab, self.num_steps)
        '''
        if one_hot:  # not using embeddign layer
            embed = tf.one_hot(self.seq, self.vocab_size)

        else:  # using embedding layer
            with tf.name_scope('embed'):
                embed_matrix = tf.get_variable('embed_matrix',
                                               shape=[self.vocab_size, self.embedding_size],
                                               initializer=tf.random_uniform_initializer())
                embed = tf.nn.embedding_lookup(embed_matrix, self.seq, name='embedding')

        self.create_actual_model(embed)

        self.get_logits()

        loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,
                                                       labels=self.label)
        self.loss = tf.reduce_sum(loss)

        self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss, global_step=self.gstep)





    def train(self):
        saver = tf.train.Saver()
        start = time.time()
        min_loss = None
        with tf.Session() as sess:
            writer = tf.summary.FileWriter('../graphs/gist', sess.graph)
            sess.run(tf.global_variables_initializer())

            ckpt = tf.train.get_checkpoint_state(os.path.dirname(config.CPT_PATH+ '/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            iteration = self.gstep.eval()
            #stream = read_data(self.path, self.vocab, self.num_steps, overlap=self.num_steps // 2)

            stream = utils.read_data_ram(self.train_index_words)
            stream_label = utils.read_label(config.DATA_PATH+config.TRAIN_LABEL_NAME)
            data= utils.read_batch(stream, self.batch_size)
            labels = utils.read_batch(stream_label,self.batch_size)

            while True:
                batch = next(data)
                label = next(labels)
                one_hoted_label = []
                for ite in label:
                    single_line = [0]*self.num_classes
                    single_line[ite]=1
                    one_hoted_label.append(single_line)


                # for batch in read_batch(read_data(DATA_PATH, vocab)):
                batch_loss, _ = sess.run([self.loss, self.opt], {self.label:one_hoted_label,self.seq: batch})
                if (iteration + 1) % self.skip_step == 0:
                    print('Iter {}. \n    Loss {}. Time {}'.format(iteration, batch_loss, time.time() - start))
                    #self.online_infer(sess)
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
        start = time.time()
        min_loss = None
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            self._check_restore_parameters(sess, saver)

            # stream = read_data(self.path, self.vocab, self.num_steps, overlap=self.num_steps // 2)

            stream = utils.read_data_ram(self.inference_index_words)
            stream_label = utils.read_label(config.DATA_PATH+config.INFERENCE_LABEL_NAME)
            data = utils.read_batch(stream, self.batch_size)
            labels = utils.read_batch(stream_label, self.batch_size)
            output_file = open(config.INFERENCE_RESULT_PATH,'a+')

            while True:

                batch = next(data)
                if len(batch) ==0:
                    break
                label = next(labels)
                one_hoted_label = []
                for ite in label:
                    single_line = [0] * self.num_classes
                    single_line[ite] = 1
                    one_hoted_label.append(single_line)

                # for batch in read_batch(read_data(DATA_PATH, vocab)):
                batch_loss, _,predicted = sess.run([self.loss, self.opt,self.label], {self.label: one_hoted_label, self.seq: batch})
                output_file.write(str(predicted))
                output_file.write('\n')
            output_file.close()





    def online_infer(self, sess):
        """ Generate sequence one character at a time, based on the previous character
        """
        for seed in ['Hillary', 'I', 'R', 'T', '@', 'N', 'M', '.', 'G', 'A', 'W']:
            sentence = seed
            state = None
            for _ in range(self.len_generated):
                batch = [utils.vocab_encode(sentence[-1], self.vocab)]
                feed = {self.seq: batch}
                if state is not None:  # for the first decoder step, the state is None
                    for i in range(len(state)):
                        feed.update({self.in_state[i]: state[i]})
                index, state = sess.run([self.sample, self.out_state], feed)
                sentence += utils.vocab_decode(index, self.vocab)
            print('\t' + sentence)
