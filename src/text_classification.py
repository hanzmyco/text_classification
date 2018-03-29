import argparse
import utils
import config
import os
from rnn import RNN
import word2vec_utils

def main():
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices={'train', 'inference'},
                        default='train', help="mode. if not specified, it's in the train mode")
    args = parser.parse_args()
    '''

    if not os.path.isdir(config.PROCESSED_PATH):
        local_dest = config.DATA_PATH
        words, vocab_size, actual_text = word2vec_utils.read_data(local_dest)
        vocab, _ = word2vec_utils.build_vocab(words, vocab_size, '../visualization')
        index_words = word2vec_utils.convert_words_to_index(actual_text, vocab, config.NUM_STEPS)


    model = config.DATA_PATH[7:len(config.DATA_PATH)-4]+'_RNN'
    utils.safe_mkdir('checkpoints')
    utils.safe_mkdir('checkpoints/' + model)

    lm = RNN(model)
    lm.vocab_size = vocab_size
    lm.index_words= index_words

    lm.create_model()




    lm.train()


if __name__ == '__main__':
    main()