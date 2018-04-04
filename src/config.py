
# parameters for processing the dataset
#train_DATA_PATH = '../data/trump_tweets.txt'
DATA_PATH = '../data/kaggle/'
TRAIN_DATA_NAME='train.txt'
TRAIN_LABEL_NAME= 'labels.txt'

INFERENCE_DATA_NAME ='train.txt'
INFERENCE_LABEL_NAME='labels.txt'
INFERENCE_RESULT_NAME='labels_result_test.txt'


OUTPUT_FILE = 'output_convo.txt'
PROCESSED_PATH = '../data/processed/'

MODEL_NAME='GRU'

CPT_PATH = '../checkpoints/'+MODEL_NAME

VOCAB_SIZE=36820

THRESHOLD = 1

PAD_ID = 0
UNK_ID = 1
START_ID = 2
EOS_ID = 3

TESTSET_SIZE = 25000

NUM_STEPS=10


CONTRACTIONS = [("i ' m ", "i 'm "), ("' d ", "'d "), ("' s ", "'s "),
				("don ' t ", "do n't "), ("didn ' t ", "did n't "), ("doesn ' t ", "does n't "),
				("can ' t ", "ca n't "), ("shouldn ' t ", "should n't "), ("wouldn ' t ", "would n't "),
				("' ve ", "'ve "), ("' re ", "'re "), ("in ' ", "in' ")]

NUM_LAYERS = 2
HIDDEN_SIZE = 256
BATCH_SIZE = 64

LR = 0.0003
MAX_GRAD_NORM = 5.0


NUM_SAMPLES = 512
ENC_VOCAB = 24133
DEC_VOCAB = 22879
NUM_CLASSES=5
EMBEDDING_SIZE=256