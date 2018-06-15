""" Prepare the data for creation of TFRecords """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import io
import csv
import numpy as np
import tensorflow as tf

# Path variables for reading in some of the data
WORD_VECS = os.getcwd() + "crawl-300d-2M.vec"
TRAIN_DATA = os.getcwd() + "train_data.csv"
VAL_DATA = os.getcwd() + "val_data.csv"

class Dataset():
    """ For reading data, processing for input, and writing to TFRecords
    """

    def sequence_to_tf_example(self, dataset, vocab):
        """ Writes the dataset examples to TFRecords using the vocab to convert sequences of tokens to IDs
        Args:
            dataset: List of paths to the dataset
            vocab: Path to the vocab file mapping words to IDs
        Writes:
            train, val = TFRecords for the train and val sets
        """

        raise NotImplementedError

    def read_data(self, path):
        """ Reads the training/validation data from the specified path
        Args:
            path: The path of the training data
        Returns:
            A list of example tuples(e.g [(sent1, label1),(sent2, label2)])
        """

        dataset = list()

        with open(path, 'rb') as f:
            reader = csv.reader(f)
            for row in reader:
                dataset.append([row[0],row[1]])

        return dataset
