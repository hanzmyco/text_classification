""" A neural chatbot using sequence to sequence model with
attentional decoder.

This is based on Google Translate Tensorflow model
https://github.com/tensorflow/models/blob/master/tutorials/rnn/translate/

Sequence to sequence model by Cho et al.(2014)

Created by Chip Huyen (chiphuyen@cs.stanford.edu)
CS20: "TensorFlow for Deep Learning Research"
cs20.stanford.edu

This file contains the code to do the pre-processing for the
Cornell Movie-Dialogs Corpus.

See readme.md for instruction on how to run the starter code.
"""
import os
import random
import re
import numpy as np
import config


def get_lines():
    id2line = {}
    file_path = config.TRAIN_DATA_PATH
    with open(file_path, 'r', errors='ignore') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            parts = line.split(' +++$+++ ')
            if len(parts) == 5:
                if parts[4][-1] == '\n':
                    parts[4] = parts[4][:-1]
                id2line[parts[0]] = parts[4]
    return id2line


def tokenize_helper(line):
    tokens = basic_tokenizer(line)
    text = ' '.join(tokens)
    for a, b in config.CONTRACTIONS:
        text = text.replace(a, b)
    return text


def tokenize_data(file_names,origin_labels,delete_repeated_labels_filename):
    print('Tokenizing the data ...')
    repeated_line_ids=set()
    deleted_labels_file = open(delete_repeated_labels_filename,'w')

    for file_name in file_names:
        seen_texts = set()
        actual_file=config.DATA_PATH+file_name

        out_file = open(config.PROCESSED_PATH+file_name+'.tok','w')
        train_lines = open(actual_file,'r').readlines()

        n = len(train_lines)
        repeated=0

        for i in range(n):
            train= train_lines[i]
            train_clean = tokenize_helper(train)
            if train_clean in seen_texts:
                print(train_clean)
                repeated += 1
                repeated_line_ids.add(i)
                continue
            seen_texts.add(train_clean)
            out_file.write(train_clean + '\n')

        print('Total repeated in', actual_file, ':', repeated)
        label_file = open(origin_labels,'r').readlines()
        for index, ite in enumerate(label_file):
            if index not in repeated_line_ids:
                deleted_labels_file.write(ite)





def prepare_dataset(questions, answers):
    # create path to store all the train & test encoder & decoder
    make_dir(config.PROCESSED_PATH)

    # random convos to create the test set
    test_ids = random.sample([i for i in range(len(questions))], config.TESTSET_SIZE)

    filenames = ['train.enc', 'train.dec', 'test.enc', 'test.dec']
    files = []
    for filename in filenames:
        files.append(open(os.path.join(config.PROCESSED_PATH, filename), 'w'))

    for i in range(len(questions)):
        if i in test_ids:
            files[2].write(questions[i] + '\n')
            files[3].write(answers[i] + '\n')
        else:
            files[0].write(questions[i] + '\n')
            files[1].write(answers[i] + '\n')

    for file in files:
        file.close()


def make_dir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass


def basic_tokenizer(line, normalize_digits=False):
    """ A basic tokenizer to tokenize text into tokens.
    Feel free to change this to suit your need. """
    line = re.sub('<u>', '', line)
    line = re.sub('</u>', '', line)
    line = re.sub('\[', '', line)
    line = re.sub('\]', '', line)
    line = line.replace('`', "'")
    words = []
    _WORD_SPLIT = re.compile("([.,!?\"'-<>:;)(])")
    _DIGIT_RE = re.compile(r"\d")
    for fragment in line.strip().lower().split():
        for token in re.split(_WORD_SPLIT, fragment):
            if not token:
                continue
            if normalize_digits:
                token = re.sub(_DIGIT_RE, '#', token)
            words.append(token)
    return words


def build_vocab(filename, normalize_digits=False):
    in_path = os.path.join(config.PROCESSED_PATH, filename)
    out_path = os.path.join(config.PROCESSED_PATH, 'vocab.{}'.format(filename[-7:-4]))

    vocab = {}
    with open(in_path, 'r') as f:
        for line in f.readlines():
            tokens = line.split()
            for token in tokens:
                if not token in vocab:
                    vocab[token] = 0
                vocab[token] += 1

    sorted_vocab = sorted(vocab, key=vocab.get, reverse=True)
    with open(out_path, 'w') as f:
        f.write('<unk>' + '\n')
        f.write('PAD'+'\n')
        index = 2
        for word in sorted_vocab:
            '''
            if vocab[word] < config.THRESHOLD:
                with open('config.py', 'a') as cf:
                    if 'enc' in filename:
                        cf.write('ENC_VOCAB = ' + str(index) + '\n')
                    else:
                        cf.write('DEC_VOCAB = ' + str(index) + '\n')
                break
            '''
            f.write(word + '\n')
            index += 1


def load_vocab(vocab_path):
    with open(vocab_path, 'r',encoding='utf-8') as f:
        words = f.read().splitlines()
    return words, {words[i]: i for i in range(len(words))}


def sentence2id(vocab, line):
    return [vocab.get(token, vocab['<unk>']) for token in line.split(' ')]


def token2id(data, mode):
    """ Convert all the tokens in the data into their corresponding
    index in the vocabulary. """
    if config.PRETRAIN_EMBD_TAG:
        vocab_path = 'vocab.embd.' + mode
        out_path = data + '.embd.' + mode + '.ids'
    else:
        vocab_path = 'vocab.' + mode
        out_path = data + '.' + mode + '.ids'

    in_path = data + '.' + mode + '.tok'

    _, vocab = load_vocab(os.path.join(config.PROCESSED_PATH, vocab_path))
    in_file = open(os.path.join(config.PROCESSED_PATH, in_path), 'r')
    out_file = open(os.path.join(config.PROCESSED_PATH, out_path), 'w')

    lines = in_file.read().splitlines()
    for line in lines:
        ids = []
        ids.extend(sentence2id(vocab, line))
        padd_input=_pad_input(ids,config.NUM_STEPS)
        out_file.write(' '.join(str(id_) for id_ in padd_input[:config.NUM_STEPS]) + '\n')


def loadGloVe(filename,vocab_tag=False,embedding=False):
    vocab = []
    embd = []
    file = open(filename,'r',encoding='utf8')

    for line in file.readlines():
        row = line.strip().split(' ')
        if vocab_tag:
            vocab.append(row[0])
        if embedding:
            embd.append([float(ite) for ite in row[1:]])
    print('Loaded GloVe!')
    file.close()
    return vocab,embd

def build_vocab_from_pretrain_embd(file_name):
    vocab,_ = loadGloVe(file_name,vocab_tag=True)
    out_file=open('../data/Processed/vocab.embd.txt','w+',encoding='utf-8')
    out_file.write('<unk>\n')
    out_file.write('PAD\n')

    for ite in vocab:
        out_file.write(ite+'\n')


def process_data(file_names):
    print('Preparing data to be model-ready ...')
    for file_name in file_names:
        if config.PRETRAIN_EMBD_TAG:
            build_vocab_from_pretrain_embd(config.PRETRAIN_EMBD_PATH)
        else:
            build_vocab(file_name)

    token2id('train', 'txt')


def load_data(enc_filename, dec_filename, max_training_size=None):
    encode_file = open(os.path.join(config.PROCESSED_PATH, enc_filename), 'r')
    decode_file = open(os.path.join(config.PROCESSED_PATH, dec_filename), 'r')
    encode, decode = encode_file.readline(), decode_file.readline()
    data_buckets = [[] for _ in config.BUCKETS]
    i = 0
    while encode and decode:
        if (i + 1) % 10000 == 0:
            print("Bucketing conversation number", i)
        encode_ids = [int(id_) for id_ in encode.split()]
        decode_ids = [int(id_) for id_ in decode.split()]
        for bucket_id, (encode_max_size, decode_max_size) in enumerate(config.BUCKETS):
            if len(encode_ids) <= encode_max_size and len(decode_ids) <= decode_max_size:
                data_buckets[bucket_id].append([encode_ids, decode_ids])
                break
        encode, decode = encode_file.readline(), decode_file.readline()
        i += 1
    return data_buckets


def _pad_input(input_, size):
    return input_ + [config.PAD_ID] * (size - len(input_))


def _reshape_batch(inputs, size, batch_size):
    """ Create batch-major inputs. Batch inputs are just re-indexed inputs
    """
    batch_inputs = []
    for length_id in range(size):
        batch_inputs.append(np.array([inputs[batch_id][length_id]
                                      for batch_id in range(batch_size)], dtype=np.int32))
    return batch_inputs


def get_batch(data_bucket, bucket_id, batch_size=1):
    """ Return one batch to feed into the model """
    # only pad to the max length of the bucket
    encoder_size, decoder_size = config.BUCKETS[bucket_id]
    encoder_inputs, decoder_inputs = [], []

    for _ in range(batch_size):
        encoder_input, decoder_input = random.choice(data_bucket)
        # pad both encoder and decoder, reverse the encoder
        encoder_inputs.append(list(reversed(_pad_input(encoder_input, encoder_size))))
        decoder_inputs.append(_pad_input(decoder_input, decoder_size))

    # now we create batch-major vectors from the data selected above.
    batch_encoder_inputs = _reshape_batch(encoder_inputs, encoder_size, batch_size)
    batch_decoder_inputs = _reshape_batch(decoder_inputs, decoder_size, batch_size)

    # create decoder_masks to be 0 for decoders that are padding.
    batch_masks = []
    for length_id in range(decoder_size):
        batch_mask = np.ones(batch_size, dtype=np.float32)
        for batch_id in range(batch_size):
            # we set mask to 0 if the corresponding target is a PAD symbol.
            # the corresponding decoder is decoder_input shifted by 1 forward.
            if length_id < decoder_size - 1:
                target = decoder_inputs[batch_id][length_id + 1]
            if length_id == decoder_size - 1 or target == config.PAD_ID:
                batch_mask[batch_id] = 0.0
        batch_masks.append(batch_mask)
    return batch_encoder_inputs, batch_decoder_inputs, batch_masks


if __name__ == '__main__':
    tokenize_data([config.TRAIN_DATA_NAME],config.DATA_PATH+config.TRAIN_LABEL_NAME,config.PROCESSED_PATH+config.TRAIN_LABEL_NAME)
    process_data([config.TRAIN_DATA_NAME+'.tok'])

