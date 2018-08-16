import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
sys.path.append('..')
import time
import tensorflow as tf
import config
import logging
import config_train_files
import config_test_files

logging.basicConfig(filename=config.LOG_PATH, level=logging.DEBUG)

def train_one_epoch(compute_graph_train, compute_graph_validation,sess,init_train,init_validate,saver, writer, epoch, iteration):
    start_time = time.time()
    sess.run(init_train)
    total_loss = 0
    n_batches = 0
    checkpoint_name = config.CPT_PATH + '/'
    total_accuracy = 0
    try:
        while True:
            batch_loss, _, accuracy = sess.run([compute_graph_train.loss, compute_graph_train.opt, compute_graph_train.acc_op])

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
    sess.run(init_validate)

    start_time = time.time()

    try:
        while True:
            batch_loss, _, accuracy = sess.run([compute_graph_validation.loss, compute_graph_validation.opt, compute_graph_validation.acc_op])
            iteration += 1
            total_loss += batch_loss
            total_accuracy += accuracy
            n_batches += 1
    except tf.errors.OutOfRangeError:
        pass

    logging.info('Average loss and accuracy on validation set at epoch {0}: {1},{2}'.format(epoch, total_loss / n_batches,
                                                                   total_accuracy / n_batches))
    print('Average loss and accuracy on validation set at epoch {0}: {1},{2}'.format(epoch, total_loss / n_batches,
                                                                   total_accuracy / n_batches))

    logging.info('Took: {0} seconds'.format(time.time() - start_time))
    print('Took: {0} seconds'.format(time.time() - start_time))

    return iteration


def train(compute_graph_train, compute_graph_validation,n_epochs):
    writer = tf.summary.FileWriter('../graphs/gist', tf.get_default_graph())

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # initilize accuracy
        running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='my_metrics')
        running_vars_initializer = tf.variables_initializer(var_list=running_vars)
        sess.run(running_vars_initializer)

        saver = tf.train.Saver(max_to_keep=3,save_relative_paths=True)
        _check_restore_parameters(sess, saver)

        iteration = compute_graph_train.gstep.eval()
        for epoch in range(n_epochs):
            iteration = train_one_epoch(compute_graph_train,compute_graph_validation,sess, saver,compute_graph_train.init,compute_graph_validation.init,writer, epoch, iteration)

    writer.close()

def _check_restore_parameters(sess, saver):
    """ Restore the previously trained parameters if there are any. """
    ckpt = tf.train.get_checkpoint_state(os.path.dirname(config.CPT_PATH + '/checkpoint'))
    if ckpt and ckpt.model_checkpoint_path:
        logging.basicConfig(filename=config.LOG_PATH, level=logging.DEBUG)
        logging.info("Loading parameters for text classifier")
        print("Loading parameters for text classifier")
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        logging.basicConfig(filename=config.LOG_PATH, level=logging.DEBUG)
        logging.info("Initializing fresh parameters for text classifier")
        print("Initializing fresh parameters for text classifier")


def inference(model):
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(model.init)

        if hasattr(config_test_files, 'INFERENCE_LABEL_NAME'):
            # initilize accuracy
            running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='my_metrics')
            running_vars_initializer = tf.variables_initializer(var_list=running_vars)
            sess.run(running_vars_initializer)

        _check_restore_parameters(sess, saver)
        output_file = open(config.PROCESSED_PATH + config_test_files.INFERENCE_RESULT_NAME, 'w+')

        try:
            while True:
                if hasattr(config_test_files, 'INFERENCE_LABEL_NAME'):
                    probability, classes, acc = sess.run(
                        [tf.nn.softmax(model.logits, name='softmax_tensor'), tf.argmax(input=model.logits, axis=1),
                         model.acc_op])
                    print(acc)
                    logging.info(str(acc))

                else:
                    probability, classes = sess.run(
                        [tf.nn.softmax(model.logits, name='softmax_tensor'), tf.argmax(input=model.logits, axis=1)])

                # print(probability)
                for ite in classes:
                    output_file.write(str(ite) + '\n')

        except tf.errors.OutOfRangeError:
            output_file.close()
            pass
