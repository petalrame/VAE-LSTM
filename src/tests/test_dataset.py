import os

import pytest
import tensorflow as tf

from src.data.dataset import Dataset
from src.data.vocab import Vocab

WORKING_DIR = os.path.abspath(os.path.dirname(__file__)) # path to file
BASE_DIR = os.path.abspath(os.path.join(WORKING_DIR, "../../data"))
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
TRAIN_RECORD = os.path.join(PROCESSED_DIR, "train.tfrecord")
VOCAB_PATH = os.path.join(PROCESSED_DIR, "vocab")

@pytest.fixture
def dataset():
    ''' Return Dataset that has loaded the vocab file '''
    vocab = Vocab(VOCAB_PATH)
    return Dataset(vocab)

def test_train_input_fn_type(dataset):
    """ Test the handler capability to create dataset from tfrecord file format
    """

    ds = dataset.train_input_fn(TRAIN_RECORD, 3)
    iterator = ds.make_initializable_iterator()

    with tf.Session() as sess:
        sess.run(iterator.initializer)
        for _ in range(1):
            features, labels = sess.run(iterator.get_next())
            print("features:", features)
            print("labels:", labels)
            assert isinstance(features, dict)
            assert isinstance(labels, dict)

def test_train_input_fn_structure(dataset):
    """ Test the handler capability to create dataset from tfrecord file format
    """

    ds = dataset.train_input_fn(TRAIN_RECORD, 3)
    iterator = ds.make_initializable_iterator()
    next = iterator.get_next()

    with tf.Session() as sess:
        sess.run(iterator.initializer)
        for _ in range(1):
            features, labels = sess.run(next)
            assert len(features["source_seq"]) == 3
            assert len(features["source_len"]) == 3
            assert len(labels["target_seq"]) == 3
            assert len(labels["target_len"]) == 3
