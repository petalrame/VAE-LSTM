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

    def dataset_to_example(self, data_dir, record_dir):
        """ Writes the dataset examples to TFRecords using the vocab to convert sequences of tokens to IDs
        Args:
            data_dir: Path to the directory containing the data
            record_dir: Path to the directory containging the tfrecords
        Returns:
            records: A list of record file paths that have been written
        """

        if not os.path.isdir(data_dir):
            raise Exception('ERROR: Path to directory does not exist or is not a directory')
            sys.exit(1)

        # find the data files
        datasets = glob.glob(os.path.join(data_dir,"*_data.csv"))
        assert len(datasets) == 2, ("ERROR: There were more than two files found") 

        records = [os.path.join(record_dir, "train.tfrecord"), os.path.join(record_dir, "val.tfrecord")]

        # Read a dataset
        for ds in datasets:
            print("Making examples for:", ds)

            data = self.read_data(ds)

            # we only update the vocab if train is set to true
            if "train_" in ds:
                record_path = records[0]
                self.vocab.train = True
                print("Saving train set to TF Record format and initializing vocab")
            elif "val_" in ds:
                record_path = records[1]
                self.vocab.train = False
                print("Saving validation set to TF Record format")

            # prepare raw text and write example to file
            with open(record_path,'w') as fp:
                writer = tf.python_io.TFRecordWriter(fp.name)
                for raw_ex in data:
                    prepped_ex = list(map(self.vocab.prep_seq, raw_ex))
                    ex = self.make_example(sequence=prepped_ex[0], target=prepped_ex[1])
                    writer.write(ex.SerializeToString())

            # we save the vocab to a location
            if ds is TRAIN_DATA:
                self.vocab.save_vocab()

            print("Finished making TFRecords for %s" % ds)

        return records



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
            "sequence": tf.FixedLenSequenceFeature([], dtype=tf.int64),
            "targets": tf.FixedLenSequenceFeature([], dtype=tf.int64)
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
        # Scalars to deflate go here
        x["sequence_len"] = tf.squeeze(["sequence_len"])
        x["target_len"] = tf.squeeze(["target_len"])

        return x

    def make_dataset(self, path, batch_size=128):
        """ Make a Tensorflow dataset that is shuffled, batched and parsed
        Args:
            path: path of the record file to unpack and read
            batch_size: Size of the batch for training
        Returns:
            A dataset that is shuffled and padded
        """

        if not os.path.isfile(path):
            raise Exception('ERROR: Path to directory does not exist or is not a directory')
            sys.exit(1)

        dataset = tf.data.TFRecordDataset([path]).map(self.parse, num_parallel_calls=5).shuffle(buffer_size=10000).map(self.expand)

        dataset = dataset.padded_batch(batch_size, padded_shapes={
            "sequence_len": 1,
            "target_len": 1,
            "sequence": tf.TensorShape([None]),
            "targets": tf.TensorShape([None])
        })

        dataset = dataset.map(self.deflate)

        return dataset

    def prep_dataset_iter(self, batch_size):
        """ Makes a dataset iterator with size batch size
        Args:
            batch_size: Size of the training batch
        Returns:
            next_element: The next element of the iterator
            training_init_op: A reinitialized training operation after each epoch
            validation_init_op: Antoher reinitialized operation for validation
        """
        train_ds = self.make_dataset(self.records[0], batch_size=batch_size)
        val_ds = self.make_dataset(self.records[1], batch_size=batch_size)

        # Make an interator object the shape of the dataset
        iterator = tf.data.Iterator.from_structure(train_ds.output_types, train_ds.output_shapes)

        next_element = iterator.get_next()
        training_init_op = iterator.make_initializer(train_ds)
        validation_init_op = iterator.make_initializer(val_ds)

        return next_element, training_init_op, validation_init_op





