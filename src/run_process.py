import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
#sys.path.append('..')
sys.path.append('utils/')
import time
import tensorflow as tf
import config
import logging
from tensorflow.python.tools import inspect_checkpoint as chkp
from tensorflow.python import pywrap_tensorflow
import numpy
import utils

def train_one_epoch(compute_graph,sess,init_train,init_validate,saver, train_writer,validation_writer, epoch, iteration,merged_summary):
    start_time = time.time()
    total_loss = 0
    n_batches = 0
    checkpoint_name = config.CPT_PATH + '/'
    total_accuracy = 0

    try:
        sess.run(init_train)
        while True:
            batch_loss, _,  accuracy,summary = sess.run([compute_graph.loss, compute_graph.opt, compute_graph.acc_op,merged_summary])
            #batch_loss, _,  accuracy = sess.run([compute_graph.loss, compute_graph.opt, compute_graph.acc_op])

            train_writer.add_summary(summary,iteration)


            iteration += 1
            total_loss += batch_loss
            total_accuracy += accuracy
            n_batches += 1

    except tf.errors.OutOfRangeError:
        pass

    saver.save(sess, checkpoint_name, iteration)

    logging.info('Average loss and accuracy at epoch {0}: {1},{2}'.format(epoch, total_loss / n_batches,
                                                                   total_accuracy / n_batches))
    print('Average loss and accuracy at epoch {0}: {1},{2}'.format(epoch, total_loss / n_batches,
                                                                   total_accuracy / n_batches))
    print('Took: {0} seconds'.format(time.time() - start_time))

    # start to test on validation
    total_loss = 0
    total_accuracy = 0
    n_batches =0
    batch_loss=0
    accuracy=0

    try:
        sess.run(init_validate)

        while True:
            #batch_loss, _, accuracy = sess.run([compute_graph.loss, compute_graph.opt,compute_graph.acc_op])
            batch_loss, _,  accuracy,summary = sess.run([compute_graph.loss, compute_graph.opt, compute_graph.acc_op,merged_summary])
            validation_writer.add_summary(summary,iteration)
            #train_writer.add_summary(summary,iteration)
            total_loss += batch_loss
            total_accuracy += accuracy
            n_batches +=1



    except tf.errors.OutOfRangeError:
        logging.info('Average loss and accuracy at validation set is {0},{1}'.format(total_loss / n_batches,total_accuracy / n_batches))
        print('Average loss and accuracy at validation set is : {0},{1}'.format(total_loss / n_batches,total_accuracy / n_batches))

    return iteration


def train(compute_graph,next_element,training_init_op,validation_init_op,n_epochs):

    with tf.Session() as sess:
        compute_graph.create_model(next_element,config.ONE_HOT_TAG,training=True)

        train_writer = tf.summary.FileWriter('../graphs/gists/'+config.PROJECT_NAME+'/'+config.MODEL_NAME+'/train', tf.get_default_graph())
        validation_writer = tf.summary.FileWriter('../graphs/gists/'+config.PROJECT_NAME+'/'+config.MODEL_NAME+'/validation', tf.get_default_graph())
        tf.summary.scalar('loss',compute_graph.loss)
        tf.summary.scalar('accuracy',compute_graph.acc_op)
        merged_summary = tf.summary.merge_all()

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        saver = tf.train.Saver(max_to_keep=3,save_relative_paths=True)
        iteration = compute_graph.gstep.eval()
        _check_restore_parameters(sess, saver)

        for epoch in range(n_epochs):
            iteration = train_one_epoch(compute_graph,sess,training_init_op,validation_init_op,saver,train_writer,validation_writer, epoch, iteration,merged_summary)
            #iteration = train_one_epoch(compute_graph,sess,training_init_op,validation_init_op,saver,train_writer, epoch, iteration,merged_summary)


        train_writer.close()
        validation_writer.close()

def _check_restore_parameters(sess, saver):
    """ Restore the previously trained parameters if there are any. """
    ckpt = tf.train.get_checkpoint_state(os.path.dirname(config.CPT_PATH + '/checkpoint'))
    logging.info(config.CPT_PATH+'/checkpoint')
    print(config.CPT_PATH+'/checkpoint')
    if ckpt and ckpt.model_checkpoint_path:
        logging.basicConfig(filename=config.LOG_PATH, level=logging.DEBUG)
        logging.info("Loading parameters for text classifier")
        print("Loading parameters for text classifier")
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        logging.basicConfig(filename=config.LOG_PATH, level=logging.DEBUG)
        logging.info("Initializing fresh parameters for text classifier")
        print("Initializing fresh parameters for text classifier")

def visualization(attention,text_data_id,f):
    f.write('<div style="margin:25px;">\n')

    for i in range(len(attention)):
        for k in range(len(attention[i])):
            f.write('<p style="margin:10px;">\n')
            for j in range(len(attention[i][k])):
                alpha = "{:.2f}".format(attention[i][k][j])
                w=text_data_id[0][i][j]
                html_text='\t<span style="margin-left:3px;background-color:rgba(255,0,0,'+str(alpha)+')">'+str(w)+'</span>\n'
                f.write(html_text)
            f.write('</p>\n')
        f.write('</div>\n')
    pass

def test_restore():
    file_name=config.CPT_PATH+config.CKPT_FILE_NAME
    reader = pywrap_tensorflow.NewCheckpointReader(file_name)
    var_to_shape_map = reader.get_variable_to_shape_map()
    for tensor_name in var_to_shape_map:
        print('tensor_name: ',tensor_name)
        target = reader.get_tensor(tensor_name)
        utils.safe_mkdir_depths(config.TF_PARAMETER_NPY_PATH+tensor_name)
        numpy.save(config.TF_PARAMETER_NPY_PATH+tensor_name,target)

def inference(compute_graph,next_element,inference_init_op):
    output_file = open(config.TEST_DATA_PATH + config.INFERENCE_RESULT_NAME, 'w+')

    with tf.Session() as sess:
        compute_graph.create_model(next_element,config.ONE_HOT_TAG,training=False)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        _check_restore_parameters(sess, saver)
        sess.run(inference_init_op)

        try:
            if config.VISUALIZATION:
                f = open(config.VISUALIZATION_PATH+'_visualization.html', 'w')
                f.write('<html style="margin:0;padding:0;"><body style="margin:0;padding:0;">\n')

            while True:
                if hasattr(config, 'TEST_LABEL_NAME'):
                    if config.SELF_ATTENTION_TAG:
                        probability, classes, acc,attention,text_data_id = sess.run(
                            [tf.nn.softmax(compute_graph.logits, name='softmax_tensor'), tf.argmax(input=compute_graph.logits, axis=1),
                             compute_graph.acc_op,compute_graph.A,next_element])
                    else:
                        probability, classes, acc= sess.run(
                            [tf.nn.softmax(compute_graph.logits, name='softmax_tensor'), tf.argmax(input=compute_graph.logits, axis=1),
                             compute_graph.acc_op])
                    print(acc)
                    logging.info(str(acc))
                    #print(acc_per_class)
                    #logging.info(str(acc_per_class))

                else:
                    if config.SELF_ATTENTION_TAG:
                        probability, classes, attention = sess.run(
                            [tf.nn.softmax(compute_graph.logits, name='softmax_tensor'), tf.argmax(input=compute_graph.logits, axis=1),compute_graph.A])
                    else:
                        probability, classes = sess.run(
                            [tf.nn.softmax(compute_graph.logits, name='softmax_tensor'), tf.argmax(input=compute_graph.logits, axis=1)])

                for ite in classes:
                    output_file.write(str(ite) + '\n')

                if config.VISUALIZATION:
                    visualization(attention,text_data_id,f)
                    pass

            if config.VISUALIZATION:
                f.write('</body></html>')
                f.close()

        except tf.errors.OutOfRangeError:
            output_file.close()
            pass
