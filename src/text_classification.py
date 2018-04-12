import argparse
import utils
import config
import os
from rnn import RNN
from cnn import CNN
import word2vec_utils
import data


def main():

    # set up check points location
    utils.safe_mkdir('../checkpoints')
    utils.safe_mkdir(config.CPT_PATH)

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices={'train', 'inference'},
                        default='train', help="mode. if not specified, it's in the train mode")
    args = parser.parse_args()

    if config.MODEL_NAME=='GRU':
        lm = RNN(config.MODEL_NAME)
    elif config.MODEL_NAME =='CNN':
        lm = CNN(config.MODEL_NAME)

    lm.vocab_size = config.VOCAB_SIZE

    if args.mode == 'train':
        if os.path.isdir(config.PROCESSED_PATH):
            local_dest = config.PROCESSED_PATH+config.TRAIN_DATA_NAME_PROCESSED
            local_dest_label = config.DATA_PATH + config.TRAIN_LABEL_NAME

            '''
                words, vocab_size, actual_text = word2vec_utils.read_data(local_dest)
                vocab, _ = word2vec_utils.build_vocab(words, vocab_size, '../visualization')
                index_words = word2vec_utils.convert_words_to_index(actual_text, vocab, config.NUM_STEPS)
                lm.train_index_words = index_words
                lm.vocab_size = vocab_size
            '''

            embd_dest = config.PRETRAIN_EMBD_PATH
            data.get_pretrain_embedding(lm,embd_dest)

            data.get_data(lm,local_dest,local_dest_label)
            lm.create_model(config.ONE_HOT_TAG)
            lm.train()

    elif args.mode == 'inference':
        if os.path.isdir(config.PROCESSED_PATH):
            local_dest = config.DATA_PATH + config.INFERENCE_DATA_NAME_PROCESSED
            words, vocab_size, actual_text = word2vec_utils.read_data(local_dest)
            vocab, _ = word2vec_utils.build_vocab(words, vocab_size, '../visualization')
            index_words = word2vec_utils.convert_words_to_index(actual_text, vocab, config.NUM_STEPS)
            lm.inference_index_words = index_words
            lm.vocab_size = vocab_size

            lm.create_model()
            lm.inference()

if __name__ == '__main__':
    main()