import os
from src.data.handler import Dataset
from src.data.vocab import Vocab

import tensorflow as tf

WORKING_DIR = os.path.abspath(os.path.dirname(__file__)) # path to file
BASE_DIR = os.path.abspath(os.path.join(WORKING_DIR, "../../data"))
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
TRAIN_RECORD = os.path.join(PROCESSED_DIR, "train.tfrecord")
VOCAB_PATH = os.path.join(PROCESSED_DIR, "vocab")

def test():
    """ Test the handler capability to create dataset from tfrecord file format
    """
    vocab_obj = Vocab(VOCAB_PATH)
    ds_handler = Dataset(vocab_obj)

    iter, next = ds_handler.prep_dataset_iter(TRAIN_RECORD, 3)

    with tf.Session() as sess:
        sess.run(iter.initializer)
        for _ in range(1):
            print(sess.run(next))

if __name__ == '__main__':
    test()
