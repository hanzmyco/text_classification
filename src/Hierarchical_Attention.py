import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
sys.path.append('..')
import tensorflow as tf
import models
import config
import rnn

class HAN(rnn.RNN):
    pass
