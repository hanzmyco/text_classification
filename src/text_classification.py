import argparse
import utils
import config
import os
from rnn import RNN
from cnn import CNN

import word2vec_utils

def main():

    # set up check points location
    model_name = config.MODEL_NAME
    utils.safe_mkdir('../checkpoints')
    utils.safe_mkdir(config.CPT_PATH)

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices={'train', 'inference'},
                        default='train', help="mode. if not specified, it's in the train mode")
    args = parser.parse_args()


    if not os.path.isdir(config.PROCESSED_PATH):
        local_dest = config.DATA_PATH
        words, vocab_size, actual_text = word2vec_utils.read_data(local_dest)
        vocab, _ = word2vec_utils.build_vocab(words, vocab_size, '../visualization')
        index_words = word2vec_utils.convert_words_to_index(actual_text, vocab, config.NUM_STEPS)


    if config.MODEL_NAME=='GRU':
        lm = RNN(config.MODEL_NAME)
    elif config.MODEL_NAME =='CNN':
        lm = CNN(config.MODEL_NAME)

    lm.vocab_size = vocab_size
    lm.index_words= index_words
    lm.create_model()

    if args.mode == 'train':
        lm.train()

    elif args.mode == 'inference':
        lm.inference()



if __name__ == '__main__':
    main()