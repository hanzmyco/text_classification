import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
sys.path.append('..')
import time
import tensorflow as tf
import utils
import word2vec_utils



class BaseModel(object):
    def base_init(self,model):
        self.model = model
        self.path = '../data/' + model + '.txt'
        self.seq = tf.placeholder(tf.int32, [None, None])
        self.label = tf.placeholder(tf.int32, [None, None])
        self.temp = tf.constant(1.5)
        self.batch_size = 10
        self.lr = 0.0003
        self.skip_step = 1
        self.len_generated = 200
        self.gstep = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        self.num_classes = 2
        self.out_state = None
        self.in_state = None
        self.sample = None
        self.num_steps = 10  # for RNN unrolled, actually use it for cut down

    def __init__(self, model):
        pass


    def create_actual_model(self, seq):
        pass

    def get_logits(self):
        pass

    def create_model(self):
        local_dest = '../data/trump_tweets_short.txt'
        words, vocab_size, actual_text = word2vec_utils.read_data(local_dest)
        self.vocab, _ = word2vec_utils.build_vocab(words, vocab_size, '../visualization')
        self.index_words = word2vec_utils.convert_words_to_index(actual_text, self.vocab, self.num_steps)

        seq = tf.one_hot(self.seq, len(self.vocab))
        self.create_actual_model(seq)
        self.get_logits()

        loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,
                                                       labels=self.label)
        self.loss = tf.reduce_sum(loss)
        # sample the next character from Maxwell-Boltzmann Distribution
        # with temperature temp. It works equally well without tf.exp
        #self.sample = tf.multinomial(tf.exp(self.logits[:, -1] / self.temp), 1)[:, 0]
        self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss, global_step=self.gstep)



    def train(self):
        saver = tf.train.Saver()
        start = time.time()
        min_loss = None
        with tf.Session() as sess:
            writer = tf.summary.FileWriter('../graphs/gist', sess.graph)
            sess.run(tf.global_variables_initializer())

            ckpt = tf.train.get_checkpoint_state(os.path.dirname('../checkpoints/' + self.model + '/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            iteration = self.gstep.eval()
            #stream = read_data(self.path, self.vocab, self.num_steps, overlap=self.num_steps // 2)

            stream = utils.read_data_ram(self.index_words)
            stream_label = utils.read_label('../data/labels.txt')
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
                    checkpoint_name = 'checkpoints/' + self.model + '/char-rnn'
                    if min_loss is None:
                        saver.save(sess, checkpoint_name, iteration)
                    elif batch_loss < min_loss:
                        saver.save(sess, checkpoint_name, iteration)
                        min_loss = batch_loss
                iteration += 1

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
