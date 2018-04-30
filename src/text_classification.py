import argparse
import utils
import config
import os
from rnn import GRU,LSTM
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
        lm = GRU(config.MODEL_NAME)
    elif config.MODEL_NAME =='LSTM':
        lm = LSTM(config.MODEL_NAME)
        print('here')
    elif config.MODEL_NAME =='CNN':
        lm = CNN(config.MODEL_NAME)
    lm.vocab_size = config.VOCAB_SIZE

    if args.mode == 'train':
        if os.path.isdir(config.PROCESSED_PATH):
    
            local_dest = config.PROCESSED_PATH+config.TRAIN_DATA_NAME_PROCESSED
            local_dest_label = config.PROCESSED_PATH + config.TRAIN_LABEL_NAME

            if config.PRETRAIN_EMBD_TAG:  # use start pretrain embd or not
                embd_dest = config.PRETRAIN_EMBD_PATH
                data.get_pretrain_embedding(lm,embd_dest)

            data.get_data(lm,local_dest,local_dest_label)
            lm.create_model(config.ONE_HOT_TAG,training=True)
            lm.train(config.EPOCH_NUM)

    elif args.mode == 'inference':
        if os.path.isdir(config.PROCESSED_PATH):
            local_dest = config.PROCESSED_PATH + config.INFERENCE_DATA_NAME_PROCESSED
            local_dest_label=None
            if hasattr(config,'INFERENCE_LABEL_NAME'):
                local_dest_label = config.PROCESSED_PATH + config.INFERENCE_LABEL_NAME

            if config.PRETRAIN_EMBD_TAG:  # use start pretrain embd or not
                embd_dest = config.PRETRAIN_EMBD_PATH
                data.get_pretrain_embedding(lm,embd_dest)

            data.get_data(lm, local_dest, local_dest_label)
            lm.create_model(config.ONE_HOT_TAG,training=False)
            lm.inference()


if __name__ == '__main__':
    main()