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

def test_prep_dataset_iter_type(dataset):
    """ Test the handler capability to create dataset from tfrecord file format
    """

    iter, next = dataset.prep_dataset_iter(TRAIN_RECORD, 3)

    with tf.Session() as sess:
        sess.run(iter.initializer)
        for _ in range(1):
            features, labels = sess.run(next)
            print("features:", features)
            print("labels:", labels)
            assert isinstance(features, dict)
            assert isinstance(labels, dict)

def test_prep_dataset_iter_structure(dataset):
    """ Test the handler capability to create dataset from tfrecord file format
    """

    iter, next = dataset.prep_dataset_iter(TRAIN_RECORD, 3)

    with tf.Session() as sess:
        sess.run(iter.initializer)
        for _ in range(1):
            features, labels = sess.run(next)
            assert len(features["source_seq"]) == 3
            assert len(features["source_len"]) == 3
            assert len(labels["target_seq"]) == 3
            assert len(labels["target_len"]) == 3
