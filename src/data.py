import tensorflow as tf
import config

def _parse_data_function(line):

    parsed_line = tf.decode_csv(line,config.READ_IN_FORMAT,field_delim=' ')
    if config.ONE_HOT_TAG:
        return tf.one_hot(tf.convert_to_tensor(parsed_line), config.VOCAB_SIZE)
    else:
        return tf.cast(parsed_line,tf.int32)
def _parse_label_function(line):

    label=tf.decode_csv(line,[[0]])
    return tf.one_hot(tf.convert_to_tensor(label),config.NUM_CLASSES)

def get_data(lm,local_dest,local_dest_label):
    dataset=tf.data.TextLineDataset(local_dest).map(_parse_data_function)
    batched_dataset = dataset.batch(config.BATCH_SIZE)
    iterator = batched_dataset.make_initializable_iterator()
    lm.seq= iterator.get_next()
    lm.train_init = iterator.make_initializer(batched_dataset)

    labelset=tf.data.TextLineDataset(local_dest_label).map(_parse_label_function)
    batched_dataset_label = labelset.batch(config.BATCH_SIZE)
    iterator_label = batched_dataset_label.make_initializable_iterator()
    lm.label = iterator_label.get_next()
    lm.train_init_label = iterator_label.make_initializer(batched_dataset_label)

