# different models for text classification task

specify your model in config.py ( now support CNN and GRU)  as MODEL_NAME

speficy the data you want to use in config.py

preprocessing your data first: get vocab and pad/ cut to same size

preprocessing :   python3 data_preprocessing.py

training: python3 text_classification.py --mode train

inference : python3 text_classification.py --mode inference

In config.py, you can specify all parameters:

DATA_PATH = '../data/kaggle/'   : data root

TRAIN_DATA_NAME= 'train.txt'    :  raw training data

TRAIN_DATA_NAME_PROCESSED= 'train.txt.ids'  :   after using preprocessing, toekns become ids

TRAIN_LABEL_NAME= 'labels.txt'  :    labels name

INFERENCE_DATA_NAME ='train.txt'

INFERENCE_DATA_NAME_PROCESSED ='train.txt.ids'

INFERENCE_LABEL_NAME='labels.txt'

INFERENCE_RESULT_NAME='labels_result_test.txt'

PROCESSED_PATH = '../data/Processed/'    : data location after processed

#MODEL_NAME='CNN'    : model names

MODEL_NAME='GRU'

CPT_PATH = '../checkpoints/'+MODEL_NAME   : check point location

VOCAB_SIZE=15217    : vocabulary size

#special token's id:

UNK_ID = 0

PAD_ID = 1

START_ID = 2

EOS_ID = 3

#unrolled number for RNN/ sentence get resize to the same length,  need to add bucketing in the future

NUM_STEPS=10

NUM_LAYERS = 2   : for stack RNN

HIDDEN_SIZE = [128,256]     : size for hidden layer

BATCH_SIZE = 256      :    batch_size

LR = 0.0003           :   learning rate


NUM_CLASSES=5        :  number of different classes for the task

ONE_HOT_TAG=False    : if ONE_HOT_TAG =True, then no embedding layer

EMBEDDING_SIZE=256   :   if you are using embedding(ONE_HOT_TAG =False), this is the embedding layer size

KERNEL_SIZE=[2,3,4]   : kernel size for CNN

NUM_FILTERS=2         :  number of filters for CNN,can be list

DROPOUT_KEEP_PROB=1.0


READ_IN_FORMAT=[[0]]*(NUM_STEPS)   #  this is for read in training data

EPOCH_NUM = 500

# using pretrain embedding or not

PRETRAIN_EMBD_TAG=False

PRETRAIN_EMBD_TRAINABLE=False

PRETRAIN_EMBD_SIZE=50

PRETRAIN_EMBD_VOCAB_SIZE = 400002

PRETRAIN_EMBD_PATH ='../data/glove.6B.50d.txt'

