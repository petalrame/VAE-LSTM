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
        self.datasets = [TRAIN_DATA, VAL_DATA] # dataset locations
        self.records = [TRAIN_RECORD, VAL_RECORD] # records for dataset
        self.batch_size = 128 # TODO: Feed in TensorFlow application flags here and read batch size from that here


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
                record_path = self.records[0]
                vocab = Vocab(train=True)
            else:
                record_path = self.records[1]
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

    def expand(self, x):
        """ Since padded_batch does not work well with scalars, we expand the scalar to vector of length 1
        Args:
            x: An example with a scalar to be expanded to legnth 1 vector
        Returns:
            x: A vector of length 1
        """

        x["sequence_len"] = tf.expand_dims(tf.convert_to_tensor(x["sequence_len"]), 0)
        x["target_len"] = tf.expand_dims(tf.convert_to_tensor(x["target_len"]), 0)

        return x

    def deflate(self, x):
        """ Since padded_batch does not work well with scalars, we squeeze the vector of length 1 back to scalar
        Args:
            x: A vector of length 1
        Returns:
            x: A scalar
        """

        x["sequence_len"] = tf.squeeze(["sequence_len"])
        x["target_len"] = tf.squeeze(["target_len"])

        return x

    def make_dataset(self, path=None, batch_size=128):
        """ Make a Tensorflow dataset that is shuffled, batched and parsed
        Args:
            path: path of the record file to unpack and read
            batch_size: Size of the batch for training
        Returns:
            A dataset that is shuffled and padded
        """

        if path is None:
            raise Exception("You must specify a path!")

        dataset = tf.data.TFRecordDataset([path]).map(self.parse, num_parallel_calls=5).shuffle(buffer_size=10000).map(self.expand)

        dataset = dataset.padded_batch(batch_size, padded_shapes={
            "sequence_len": 1,
            "target_len": 1,
            "sequence": tf.TensorShape([None]),
            "targets": tf.TensorShape([None])
        })

        dataset = dataset.map(self.deflate)

        return dataset

    def prep_dataset_iter(self, batch_size=128):
        """ Makes a dataset iterator with size batch size
        Args:
            batch_size: Size of the training batch
        Returns:
            next_element: The next element of the iterator
            training_init_op: A reinitialized training operation after each epoch
            validation_init_op: Antoher reinitialized operation for validation
        """
        train_ds = self.make_dataset(self.records[0], batch_size=self.batch_size)
        val_ds = self.make_dataset(self.records[1], batch_size=self.batch_size)

        # Make an interator object the shape of the dataset
        iterator = tf.data.Iterator.from_structure(train_ds.output_types, train_ds.output_shapes)

        next_element = iterator.get_next()
        training_init_op = iterator.make_initializer(train_ds)
        validation_init_op = iterator.make_initializer(val_ds)

        return next_element, training_init_op, validation_init_op





