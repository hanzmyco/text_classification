import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
sys.path.append('..')
import time
import tensorflow as tf
import config
import logging



logging.basicConfig(filename=config.LOG_PATH, level=logging.DEBUG)

def train_one_epoch(compute_graph,sess,init_train,init_validate,saver, writer, epoch, iteration):
    start_time = time.time()
    total_loss = 0
    n_batches = 0
    checkpoint_name = config.CPT_PATH + '/'
    total_accuracy = 0

    try:
        sess.run(init_train)
        while True:
            batch_loss, _, accuracy = sess.run([compute_graph.loss, compute_graph.opt, compute_graph.acc_op])

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
            batch_loss, _, accuracy = sess.run([compute_graph.loss, compute_graph.opt, compute_graph.acc_op])
            total_loss += batch_loss
            total_accuracy += accuracy
            n_batches +=1

    except tf.errors.OutOfRangeError:
        logging.info('Average loss and accuracy at validation set is {0},{1}'.format(total_loss / n_batches,total_accuracy / n_batches))
        print('Average loss and accuracy at validation set is : {0},{1}'.format(total_loss / n_batches,total_accuracy / n_batches))
        pass


    return iteration


def train(compute_graph,next_element,training_init_op,validation_init_op,n_epochs):
    writer = tf.summary.FileWriter('../graphs/gist', tf.get_default_graph())

    with tf.Session() as sess:
        compute_graph.create_model(next_element,config.ONE_HOT_TAG,training=True)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        # initilize accuracy
        #running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='my_metrics')
        #running_vars_initializer = tf.variables_initializer(var_list=running_vars)
        #sess.run(running_vars_initializer)

        saver = tf.train.Saver(max_to_keep=3,save_relative_paths=True)


        iteration = compute_graph.gstep.eval()

        _check_restore_parameters(sess, saver)

        for epoch in range(n_epochs):


            iteration = train_one_epoch(compute_graph,sess,training_init_op,validation_init_op,saver,writer, epoch, iteration)

    writer.close()

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


def inference(compute_graph,next_element,inference_init_op):


    output_file = open(config.TEST_DATA_PATH + config.INFERENCE_RESULT_NAME, 'w+')

    with tf.Session() as sess:
        compute_graph.create_model(next_element,config.ONE_HOT_TAG,training=False)

        saver = tf.train.Saver()

        sess.run(tf.global_variables_initializer())

        sess.run(tf.local_variables_initializer())

        _check_restore_parameters(sess, saver)


        sess.run(inference_init_op)

        '''
        if hasattr(config_test_files, 'INFERENCE_LABEL_NAME'):
            # initilize accuracy
            running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='my_metrics')
            running_vars_initializer = tf.variables_initializer(var_list=running_vars)
            sess.run(running_vars_initializer)
        '''


        try:
            while True:
                if hasattr(config, 'TEST_LABEL_NAME'):
                    probability, classes, acc = sess.run(
                        [tf.nn.softmax(compute_graph.logits, name='softmax_tensor'), tf.argmax(input=compute_graph.logits, axis=1),
                         compute_graph.acc_op])
                    print(acc)
                    logging.info(str(acc))

                else:
                    probability, classes = sess.run(
                        [tf.nn.softmax(compute_graph.logits, name='softmax_tensor'), tf.argmax(input=compute_graph.logits, axis=1)])

                # print(probability)
                for ite in classes:
                    output_file.write(str(ite) + '\n')

        except tf.errors.OutOfRangeError:
            output_file.close()
            pass
