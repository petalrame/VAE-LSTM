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
        decoder_tgt = target[1:] # original target seq with appended EOS token 
        target = target[:-1] # original target seq with prepended start token
        target_len = len(target) # decoder_tgt and target have the same size

        # create key/value pair for features
        context_features = {
            "source_len": _int64_feature(seq_len),
            "target_len": _int64_feature(target_len),
        }
        feature_list = {
            "source_seq": _int64_feature_list(sequence),
            "target_seq": _int64_feature_list(target),
            "decoder_tgt": _int64_feature_list(decoder_tgt)
        }

        return tf.train.SequenceExample(context=tf.train.Features(feature=context_features), feature_lists=tf.train.FeatureLists(feature_list=feature_list))

    def train_input_fn(self, path, batch_size):
        """ Make a Tensorflow dataset that is shuffled, batched and parsed
        Args:
            path: path of the record file to unpack and read
            batch_size: Size of the batch for training
        Returns:
            A dataset that is shuffled and padded
        """

        if not os.path.isfile(path):
            raise Exception('ERROR: Provided path is not a file')

        def _parse(ex):
            """ Explain to TF how to go back from a serialized example to tensors
            Args:
                ex: An example
            Returns:
                A dictionary of tensors
            """
            # Define how to parse the example
            context_features = {
                "source_len": tf.FixedLenFeature([], dtype=tf.int64),
                "target_len": tf.FixedLenFeature([], dtype=tf.int64)
            }
            sequence_features = {
                "source_seq": tf.FixedLenSequenceFeature([], dtype=tf.int64),
                "target_seq": tf.FixedLenSequenceFeature([], dtype=tf.int64),
                "decoder_tgt": tf.FixedLenSequenceFeature([], dtype=tf.int64)
            }
            #Parse the example and return dict of tensors
            context_parsed, sequence_parsed = tf.parse_single_sequence_example(
                serialized=ex,
                context_features=context_features,
                sequence_features=sequence_features
            )

            return {"source_seq": sequence_parsed["source_seq"],
                    "source_len": context_parsed["source_len"]}, {"target_seq": sequence_parsed["target_seq"],
                                                                  "target_len": context_parsed["target_len"],
                                                                  "decoder_tgt": sequence_parsed["decoder_tgt"]}

        dataset = tf.data.TFRecordDataset([path], num_parallel_reads=4).map(_parse, num_parallel_calls=10).shuffle(buffer_size=2*batch_size+1).repeat(None)

        padded_shapes = ({"source_seq": tf.TensorShape([None]), # pads to largest sentence in batch
            "source_len": tf.TensorShape([])}, # No padding
            {"target_seq": tf.TensorShape([None]),
            "target_len": tf.TensorShape([]),
            "decoder_tgt": tf.TensorShape([None])})

        dataset = dataset.padded_batch(batch_size, padded_shapes=padded_shapes)

        # enables pipelines
        dataset = dataset.prefetch(2)

        return dataset

    def predict_input_fn(self, path, batch_size=1):
        """ Used to shape input for predict mode
        Args:
            path: Path to the data to be read
            batch_size: Optional batch size for prediction
        Returns:
            dataset: A tf.data.Dataset where the tuple returned is features, _
        """
        if path is not None:
            assert batch_size is not None, "Error! If path is provided, batch size must be too!"

        if not os.path.isfile(path):
            raise Exception("Error! The path provided is not a file.")

        def _preprocess(ex):
            """ Preprocesses the input
            """
            parsed = self.vocab.prep_seq(ex)
            ex_len = len(parsed)
            return parsed, ex_len

        # create the dataset and preprocess it.
        seq_input = []
        seq_len = []
        with open(path) as file:
            lines = file.readlines()
            for idx, row in enumerate(lines):
                if idx == 0:
                    # skip header row 
                    continue
                else:
                    parsed = self.vocab.prep_seq(row)
                    seq_input.append(parsed)
                    seq_len.append(len(parsed))
            assert len(seq_input) == len(seq_len), "Error! seq_input must be the same length as the seq_len"

        # pad the data
        max_len = max(seq_len)
        for idx, _ in enumerate(seq_input):
            seq_input[idx].extend([0]*(max_len-seq_len[idx]))

        # preprocess data
        seq_input = np.array([np.asarray(x, dtype=np.int64) for x in seq_input])
        seq_len = np.array([np.asarray(x, dtype=np.int64) for x in seq_len])

        return tf.estimator.inputs.numpy_input_fn(
            x={"source_seq": seq_input, "source_len": seq_len},
            batch_size=batch_size,
            shuffle=False
        )





