""" Prepare the data for creation of TFRecords """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import io
import csv
import numpy as np
import tensorflow as tf

class Dataset(object):
    """ For reading data, processing for input, and writing to TFRecords
    """

    def __init__(self, vocab):
        self.vocab = vocab

    def dataset_to_example(self, data_path, record_path):
        """ Writes the dataset examples to TFRecords using the vocab to convert sequences of tokens to IDs
        Args:
            data_path: Path to the file containing the data
            record_path: Path to save the tfRecord file(s) to
        Returns:
            records: A list of record file paths that have been written
        """

        if not os.path.isfile(data_path):
            raise Exception('ERROR: Path to directory does not exist or is not a directory')

        print("Reading data located at:", data_path)

        # Read a dataset
        data = self._read_csv_data(data_path)

        print("Processing {0} and writing to: {1}".format(data_path, record_path))

        # prepare raw text and write example to file
        with open(record_path,'w') as fp:
            writer = tf.python_io.TFRecordWriter(fp.name)
            for raw_ex in data:
                prepped_ex = list(map(self.vocab.prep_seq, raw_ex))
                ex = self._make_example(sequence=prepped_ex[0], target=prepped_ex[1])
                writer.write(ex.SerializeToString())

        print("Finished making TFRecords for %s" % data_path)

        return

    def _read_csv_data(self, path):
        """ Reads the training/validation data from the specified path
        Args:
            path: The path of the training data
        Returns:
            A list of examples(e.g [[sent1, label1],[sent2, label2]]) where each example is a list of [sent, label]
            where sent and label is a text string
        """

        dataset = list()

        with open(path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                dataset.append([row[0],row[1]])

        return dataset

    def _make_example(self, sequence, target):
        """ Returns a SequenceExample for the given inputs and labels
        Args:
            sequence: A list of input IDs.
            target: A list of target IDs
        Returns:
            A tf.train.SequenceExample containing inputs and targets
        """
        # Convert to feature
        def _int64_feature(value):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

        # Convert list of IDs to int feature list
        def _int64_feature_list(values):
            return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])

        # calculate lengths
        seq_len = len(sequence)
        target_len = len(target)

        # create key/value pair for features
        context_features = {
            "sequence_len": _int64_feature(seq_len),
            "target_len": _int64_feature(target_len),
        }
        feature_list = {
            "sequence": _int64_feature_list(sequence),
            "targets": _int64_feature_list(target),
        }

        return tf.train.SequenceExample(context=tf.train.Features(feature=context_features), feature_lists=tf.train.FeatureLists(feature_list=feature_list))

    def make_dataset(self, path, batch_size):
        """ Make a Tensorflow dataset that is shuffled, batched and parsed
        Args:
            path: path of the record file to unpack and read
            batch_size: Size of the batch for training
        Returns:
            A dataset that is shuffled and padded
        """

        if not os.path.isfile(path):
            raise Exception('ERROR: Path to directory does not exist or is not a directory')

        def _parse(ex):
            """ Explain to TF how to go back from a serialized example to tensors
            Args:
                ex: An example
            Returns:
                A dictionary of tensors
            """
            # Define how to parse the example
            context_features = {
                "sequence_len": tf.FixedLenFeature([], dtype=tf.int64),
                "target_len": tf.FixedLenFeature([], dtype=tf.int64)
            }
            sequence_features = {
                "sequence": tf.FixedLenSequenceFeature([], dtype=tf.int64),
                "targets": tf.FixedLenSequenceFeature([], dtype=tf.int64)
            }
            #Parse the example and return dict of tensors
            context_parsed, sequence_parsed = tf.parse_single_sequence_example(
                serialized=ex,
                context_features=context_features,
                sequence_features=sequence_features
            )

            return {"sequence": sequence_parsed["sequence"],
                    "sequence_len": context_parsed["sequence_len"], "target_len": context_parsed["target_len"]}, sequence_parsed["targets"]

        dataset = tf.data.TFRecordDataset([path]).map(_parse, num_parallel_calls=5).shuffle(buffer_size=2*batch_size+1)

        dataset = dataset.padded_batch(batch_size, padded_shapes=({
            "sequence": [None],
            "sequence_len": [],
            "target_len": []},
            tf.TensorShape([None]))
        )

        return dataset

    def prep_dataset_iter(self, path, batch_size):
        """ Makes a dataset iterator with size batch size
        Args:
            path: path to the tfrecord file
            batch_size: Size of the training batch
        Returns:
            iterator: The iterator for the dataset. To be initialized with iterator.initializer
            next_element: The next element of the iterator
        """
        if not os.path.isfile(path):
            raise Exception("ERROR: Provided path is not a file")

        ds = self.make_dataset(path, batch_size=batch_size)

        # Make an interator object the shape of the dataset
        iterator = ds.make_initializable_iterator()
        next_element = iterator.get_next()

        return iterator, next_element





