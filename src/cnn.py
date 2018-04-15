import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
sys.path.append('..')
import tensorflow as tf
import models
import config



class CNN(models.BaseModel):
    def __init__(self,model):
        self.base_init(model)
        self.kernel_sizes=[2,3,4]
        self.num_filters=2
        self.dropout_keep_prob = 1.0


    def create_actual_model(self, embd):
        if config.ONE_HOT_TAG:
            self.kernel_width = self.vocab_size
        else:
            self.kernel_width = self.embedding_size
        pooled_outputs=[]
        for i,kernel_size in enumerate(self.kernel_sizes):
            with tf.name_scope("cnn_max_pool-%s" % kernel_size):
                kernel_shape=[kernel_size,self.kernel_width,1,self.num_filters]
                W = tf.Variable(tf.truncated_normal(kernel_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")
                conv = tf.nn.conv2d(tf.expand_dims(embd,-1),W,strides=[1,1,1,1],padding='VALID',name='conv')
                h = tf.nn.relu(tf.nn.bias_add(conv,b),name='relu')
                pooled = tf.nn.max_pool(h,ksize=[1,self.num_steps-kernel_size+1,1,1],strides=[1,1,1,1],padding='VALID',name='max_pooled')
                pooled_outputs.append(pooled)
        num_filters_total = self.num_filters*len(self.kernel_sizes)
        h_pool = tf.concat(pooled_outputs,3)

        with tf.name_scope('dropouts'):
            h_pool_drop = tf.nn.dropout(tf.reshape(h_pool,[-1,num_filters_total]),self.dropout_keep_prob)
        self.output = h_pool_drop

    def get_logits(self):
        self.logits = tf.layers.dense(self.output, self.num_classes, None)




