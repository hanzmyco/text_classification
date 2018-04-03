import os
import gzip
import shutil
import struct
import urllib
import random

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

#from matplotlib import pyplot as plt
#import numpy as np
#import tensorflow as tf


def safe_mkdir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass



def vocab_encode(text, vocab):
    return [vocab.index(x) + 1 for x in text if x in vocab]


def vocab_decode(array, vocab):
    return ''.join([vocab[x - 1] for x in array])


def read_data(filename, vocab, window, overlap):
    lines = [line.strip() for line in open(filename, 'r').readlines()]
    while True:
        random.shuffle(lines)

        for text in lines:
            text = vocab_encode(text, vocab)
            for start in range(0, len(text) - window, overlap):
                chunk = text[start: start + window]
                chunk += [0] * (window - len(chunk))
                yield chunk


def read_data_ram(index_words):
    #while True:
        for sentence in index_words:
            yield sentence

def read_label(labels_file):
    while True:
        for line in open(labels_file):
            label = int(line.strip())
            yield label

def read_batch(stream, batch_size):
    batch = []
    for element in stream:
        batch.append(element)
        if len(batch) == batch_size:
            yield batch
            batch = []
    yield batch
