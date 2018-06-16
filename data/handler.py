""" Prepare the data for creation of TFRecords """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import io
import csv
import numpy as np
import tensorflow as tf
from vocab import Vocab

# Path variables for reading in some of the data
TRAIN_DATA = os.getcwd() + "train_data.csv"
VAL_DATA = os.getcwd() + "val_data.csv"
TRAIN_RECORD = os.getcwd() + "train.tfrecord"
VAL_RECORD = os.getcwd() + "val.tfrecord"

class Dataset():
    """ For reading data, processing for input, and writing to TFRecords
    """

    def __init__(self):
        self.datasets = [TRAIN_DATA, VAL_DATA]


    def dataset_to_example(self):
        """ Writes the dataset examples to TFRecords using the vocab to convert sequences of tokens to IDs
        Writes:
            train, val = TFRecords for the train and val sets
        """

        # Read the dataset
        for ds in self.datasets:
            print("Making examples for:", ds)

            data = self.read_data(ds)
            record_path = None

            if ds is TRAIN_DATA:
                record_path = TRAIN_RECORD
                vocab = Vocab(train=True)
            else:
                record_path = VAL_RECORD
                vocab = Vocab(train=False)

            print("Saving dataset to TF example and initializing vocab")

            with open(record_path,'w') as fp:
                writer = tf.python_io.TFRecordWriter(fp.name)
                for raw_ex in data:
                    prepped_ex = list(map(vocab.prep_seq, raw_ex))
                    ex = self.make_example(sequence=prepped_ex[0], target=prepped_ex[1])
                    writer.write(ex.SerializeToString())

            if ds is TRAIN_DATA:
                vocab.save_vocab()

            print("Finished making TFRecords for %s" % ds)



    def make_example(self, sequence=list(), target=list()):
        """ Makes an example for a list
        Args:
            sequence: A list of IDs that represents a sequence
            target: A target list of IDs for the model(also representing a sequence)
        Returns:
            ex: A TFExample of the example fed in
        """

        ex = tf.train.SequenceExample()

        # Adding non-sequential features
        sequence_len = len(sequence)
        target_len = len(target)
        ex.context.feature["sequence_len"].int64_list.value.append(sequence_len)
        ex.context.feature["target_len"].int64_list.value.append(target_len)

        # Add feature lists for the two sequential features in the example
        fl_sequence = ex.feature_lists.feature_list["sequence"]
        fl_targets = ex.feature_lists.feature_list["targets"]
        
        for token, target in zip(sequence, target):
            fl_sequence.feature_add().int64_list.value.append(token)
            fl_targets.feature_add().int64_list.value.append(target)
        
        return ex

    def read_data(self, path):
        """ Reads the training/validation data from the specified path
        Args:
            path: The path of the training data
        Returns:
            A list of examples(e.g [[sent1, label1],[sent2, label2]]) where each example is a list of [sent, label]
        """

        dataset = list()

        with open(path, 'rb') as f:
            reader = csv.reader(f)
            for row in reader:
                dataset.append([row[0],row[1]])

        return dataset

    @staticmethod
    def parse(ex):
        """ Explain to TF how to go back from a serialized example to tensors
        Args:
            ex: An example
        Returns:
            A dictionary of tensors
        """

        # Define how to prase the example
        context_features = {
            "sequence_len": tf.FixedLenFeature([], dtype=tf.int64),
            "target_len": tf.FixedLenFeature([], dtype=tf.int64)
        }

        sequence_features = {
            "sequence": tf.FixedLenFeature([], dtype=tf.int64),
            "targets": tf.FixedLenFeature([], dtype=tf.int64)
        }

        #Parse the example and return dict of tensors
        context_parsed, sequence_parsed = tf.parse_single_sequence_example(
            serialized=ex,
            context_features=context_features,
            sequence_features=sequence_features
        )

        return {"source": sequence_parsed["sequence"], "target": sequence_parsed["targets"], "source_len": context_parsed["sequence_len"], "target_len": context_parsed["target_len"]}
