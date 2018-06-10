""" Prepare the data for creation of TFRecords """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict, Counter 
import os
import pickle # TODO: we might need to remove this at a later time, pickle may not 100% be needed
import numpy as np
import tensorflow as tf

# Additions of special characters for sequence generation. You may add your own...
PAD = "<PAD>"
START = "<START>"
EOS = "<EOS>"

# Path variables for reading in some of the data
WORD_VECS = os.getcwd() + "crawl-300d-2M.vec"
TRAIN_DATA = os.getcwd() + "train_data.csv"
VAL_DATA = os.getcwd() + "val_data.csv"

class GenPrep():
    """ For reading data, processing for input, and writing to TFRecords
    """
    def __init__(self, tokenizer_fn):
        self.vocab = defaultdict(self.next_val) # maps tokens to ids. Autogenerate next id as needed
        self.reverse_vocab = {}
        self.embeddings = {} # map IDs to vector embeddings
        self.token_counter = Counter() # counts token frequency
        # Add special characters to the vocab first
        self.vocab[0] = PAD
        self.vocab[1] = START
        self.vocab[2] = EOS
        self.next = 2 # after 2 is 3 and so on...
        self.tokenizer = tokenizer_fn
        self.embedding_matrix = self.read_embeddings()

    def next_val(self):
        self.next += 1
        return self.next

    def training_prep():
        """ Reads in data for training/validation, tokenizes it, then makes a TF example from it.
        """

        raise Exception('The training_prep method must be overwritten')

    def sequence_to_tf_example(self, input_ex):
        """ Coverts a sequence(a sentence) to tf.example
        Args:
            input_ex: An array of values that constitute one example to be written to tf.Example(NOTE: Preprocessing and cleaning must be done beforehand)
            e.g [input_sent1, input_sent2, label_sent]
        Returns:
            ex: A sequence example
        """

        raise NotImplementedError

    def prep_seq(self, seq):
        """ Tokenizes/cleans and converts sequence of text to sequence of (chars or words)
        Args:
            seq: A sequence to to be prepared
        Returns:
            A list of IDs
        """

        raise NotImplementedError

    def tokenize(self, seq):
        """ Tokenizes the input sequence.
        Args:
            seq: A sequence to be tokenized/cleaned(e.g "Hello, this is a sequence.")
        Returns: 
            A list of tokens(e.g words or charcters)
        """

        raise NotImplementedError

    def map_to_ids(self, tok_seq):
        """ Maps a list of tokens to their respective ids
        Args:
            tok_seq: A sequence of tokens(words or chars)
        Returns:
            A list of IDs
        """

        raise NotImplementedError

    def read_embeddings(self, path=None):
        """ Reads and stores the FastText word embeddings from a file to a python dictionary with word as the key and vector as the value.
        Args:
            path: path to the .vec file
        Returns:
            embedding_matrix: dict containing word/vector pair
        """

        if path is None:
            return {}

        raise NotImplementedError

    def read_data(self, path):
        """ Reads the training data from the specified path
        Args:
            path: The path of the training data
        Returns:
            A list of example tuples(e.g [(sent1, label1),(sent2, label2)])
        """

        raise Exception('The read_data method should be overwritten')

    def ids_to_text(self, seq):
        """ Maps a sequence of IDs to a string
        Args:
            seq: A sequence of IDs
        Returns:
            text: A text string that is supposed to be a sentence
        """

        raise NotImplementedError
