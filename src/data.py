import tensorflow as tf
import config
import data_preprocessing

def _parse_data_function(line):
    parsed_line = tf.decode_csv(line,config.READ_IN_FORMAT,field_delim=' ')
    if config.ONE_HOT_TAG:
        return tf.one_hot(tf.convert_to_tensor(parsed_line), config.VOCAB_SIZE)
    else:
        return tf.cast(parsed_line,tf.int32)

def _test_parse_data_function(line):
    parsed_line =tf.decode_raw(line,out_type = tf.unit8)
    return tf.cast(parsed_line,tf.int32)


def _parse_label_function(line):

    label=tf.decode_csv(line,[[0]])
    return tf.one_hot(tf.convert_to_tensor(label),config.NUM_CLASSES)

def get_data(local_dest,local_dest_label=None,iterator = None):
    dataset=tf.data.TextLineDataset(local_dest).map(_parse_data_function)

    if local_dest_label:
        labelset=tf.data.TextLineDataset(local_dest_label).map(_parse_label_function)
        batched_dataset = tf.data.Dataset.zip((dataset,labelset)).batch(config.BATCH_SIZE)
    else:
        batched_dataset = dataset.batch(config.BATCH_SIZE)

    if iterator == None:
        iterator = tf.data.Iterator.from_structure(batched_dataset.output_types,batched_dataset.output_shapes)
    init_op = iterator.make_initializer(batched_dataset)

    return iterator,init_op


def get_pretrain_embedding(lm,local_dest):
    _, embd = data_preprocessing.loadGloVe(local_dest,embedding=True)
    embd.insert(0, [1] * config.PRETRAIN_EMBD_SIZE)
    embd.insert(0, [0] * config.PRETRAIN_EMBD_SIZE)
    lm.embedding_size=config.PRETRAIN_EMBD_SIZE
    lm.pretrain_embd=tf.convert_to_tensor(embd)
    lm.vocab_size=config.PRETRAIN_EMBD_VOCAB_SIZE
