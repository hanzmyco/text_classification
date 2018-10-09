import argparse
import utils
import config
import os
from LSTM import LSTM
from cnn import CNN
from GRU import GRU
import data
import run_process
import logging
import tensorflow as tf

def main():
    # set up check points location
    utils.safe_mkdir_depths('../checkpoints/'+config.PROJECT_NAME+'/'+config.MODEL_NAME)
    '''
    utils.safe_mkdir('../checkpoints')
    utils.safe_mkdir('../checkpoints/'+config.PROJECT_NAME)
    utils.safe_mkdir('../checkpoints/'+config.PROJECT_NAME+'/'+config.MODEL_NAME)
    '''

    '''
    utils.safe_mkdir('../log')
    utils.safe_mkdir('../log/'+config.PROJECT_NAME)
    utils.safe_mkdir('../log/'+config.PROJECT_NAME+'/'+config.MODEL_NAME)
    '''
    utils.safe_mkdir_depths('../log/'+config.PROJECT_NAME+'/'+config.MODEL_NAME)
    logging.basicConfig(filename=config.LOG_PATH,level=logging.DEBUG)

    '''
    utils.safe_mkdir('../visualization')
    utils.safe_mkdir('../visualization/'+config.PROJECT_NAME)
    utils.safe_mkdir('../visualization/'+config.PROJECT_NAME+'/'+config.MODEL_NAME)
    '''
    utils.safe_mkdir_depths('../visualization/'+config.PROJECT_NAME+'/'+config.MODEL_NAME)


    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices={'train', 'inference','transfer'},
                        default='train', help="mode. if not specified, it's in the train mode")
    args = parser.parse_args()

    if config.MODEL_NAME=='GRU':
        compute_graph = GRU(config.MODEL_NAME)
    elif config.MODEL_NAME =='LSTM':
        compute_graph = LSTM(config.MODEL_NAME)
    elif config.MODEL_NAME =='CNN':
        compute_graph = CNN(config.MODEL_NAME)

    compute_graph.vocab_size = config.VOCAB_SIZE

    if args.mode == 'train':
        local_dest = config.TRAIN_DATA_PATH+config.TRAIN_DATA_NAME
        local_dest_label = config.TRAIN_DATA_PATH+config.TRAIN_LABEL_NAME
        validation_dest=config.VALIDATION_DATA_PATH+config.VALIDATION_DATA_NAME
        validation_dest_label=config.VALIDATION_DATA_PATH+config.VALIDATION_LABEL

        if config.PRETRAIN_EMBD_TAG:  # use start pretrain embd or not
            embd_dest = config.PRETRAIN_EMBD_PATH
            data.get_pretrain_embedding(compute_graph,embd_dest)

        iterator,training_init_op= data.get_data(local_dest,local_dest_label)
        next_element=iterator.get_next()
        _,validation_init_op =data.get_data(validation_dest,validation_dest_label,iterator)
        run_process.train(compute_graph,next_element,training_init_op,validation_init_op,config.EPOCH_NUM)

    elif args.mode == 'inference':
        local_dest = config.TEST_DATA_PATH + config.TEST_DATA_NAME
        local_dest_label=None
        if hasattr(config,'TEST_LABEL_NAME'):
            local_dest_label = config.TEST_DATA_PATH + config.TEST_LABEL_NAME

        if config.PRETRAIN_EMBD_TAG:  # use start pretrain embd or not
            embd_dest = config.PRETRAIN_EMBD_PATH
            data.get_pretrain_embedding(compute_graph,embd_dest)

        iterator,inference_init_op = data.get_data(local_dest, local_dest_label)
        next_element=iterator.get_next()
        run_process.inference(compute_graph,next_element,inference_init_op)

    elif args.mode == 'transfer':
        run_process.test_restore()


if __name__ == '__main__':
    main()
