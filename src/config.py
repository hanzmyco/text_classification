
# parameters for processing the dataset
DATA_PATH = '../data/video_info/10_cross_validation/'
TRAIN_DATA_NAME= 'train/train.txt'
TRAIN_DATA_NAME_PROCESSED= 'train/train.txt.ids.0'
TRAIN_LABEL_NAME= 'train/label.txt.0'

#INFERENCE_DATA_NAME ='train.txt'
INFERENCE_DATA_NAME_PROCESSED ='test/test.txt.ids.0'
INFERENCE_LABEL_NAME='labels.txt'
INFERENCE_RESULT_NAME='labels_result_test.txt'

PROCESSED_PATH = '../data/video_info/'

LOG_PATH='../log/video_info/10_cross_validation_0/'

#MODEL_NAME='CNN'
MODEL_NAME='GRU'
#MODEL_NAME='LSTM'


CPT_PATH = '../model/video_info/checkpoints/'+MODEL_NAME+'/0'

VOCAB_SIZE=48301
UNK_ID = 0
PAD_ID = 1
START_ID = 2
EOS_ID = 3



NUM_STEPS=10
NUM_LAYERS = 2
HIDDEN_SIZE = [128,256]
BATCH_SIZE = 64
LR = 0.0003
#NUM_SAMPLES = 512
NUM_CLASSES=8
EMBEDDING_SIZE=256
KERNEL_SIZE=[2,3,4]
NUM_FILTERS=2
DROPOUT_KEEP_PROB=1.0

READ_IN_FORMAT=[[0]]*(NUM_STEPS)
ONE_HOT_TAG=False
EPOCH_NUM = 1000

PRETRAIN_EMBD_TAG=False
PRETRAIN_EMBD_TRAINABLE=False
PRETRAIN_EMBD_SIZE=50
PRETRAIN_EMBD_VOCAB_SIZE = 400002
PRETRAIN_EMBD_PATH ='../data/glove.6B.50d.txt'

SELF_ATTENTION_TAG = False
ATTENTION_SIZE = 15
NUM_TOPICS =4
ATTENTION_COEF=0.004

HIERACHICAL_ATTENTION_TAG=False

MODEL_BI_DIRECTION = True
