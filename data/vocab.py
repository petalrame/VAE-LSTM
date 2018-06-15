""" Prepare the data for creation of TFRecords """
from __future__ import absolute_import, division, print_function

import io
import os
import pickle
from collections import Counter, defaultdict

import nltk
import numpy as np
from gensim.models import FastText as ft

# Additions of special characters for sequence generation. You may add your own...
PAD = "<PAD>"
START = "<START>"
EOS = "<EOS>"
UNK = "<UNK>"

# Path variables for reading in some of the data
WORD_VECS = os.getcwd() + "crawl-300d-2M.vec"
TRAIN_DATA = os.getcwd() + "train_data.csv"
VAL_DATA = os.getcwd() + "val_data.csv"

class Vocab():
    """ For reading data, processing for input, and writing to TFRecords
    """
    def __init__(self, tokenizer_fn):
        self.vocab = defaultdict(self.next_val) # maps tokens to ids. Autogenerate next id as needed
        self.reverse_vocab = {}
        self.embeddings = {} # map of ID to vecctor
        self.token_counter = Counter() # counts token frequency
        # Add special characters to the vocab first
        self.vocab[PAD] = 0
        self.vocab[START] = 1
        self.vocab[EOS] = 2
        self.vocab[UNK] = 3
        self.next = 3 # after 2 is 3 and so on...
        self.tokenizer = tokenizer_fn

    def next_val(self):
        self.next += 1
        return self.next

    def prep_seq(self, seq):
        """ Tokenizes/cleans and converts sequence of text to sequence of (chars or words)
        Args:
            seq: A sequence to to be prepared
        Returns:
            A list of IDs
        """

        seq = self.tokenize(seq)

        seq = self.map_to_ids(seq)

        return seq

    def tokenize(self, seq):
        """ Tokenizes the input sequence.
        Args:
            seq: A sequence to be tokenized/cleaned(e.g "Hello, this is a sequence.")
        Returns: 
            A list of tokens(e.g words or charcters)
        """

        seq = seq.lower()

        return nltk.word_tokenize(seq)

    def map_to_ids(self, tok_seq, train=False):
        """ Maps a list of tokens to their respective ids
        Args:
            tok_seq: A sequence of tokens(words or chars)
            train: Determines whether to add the tokens to the vocab
        Returns:
            A list of IDs
        """

        return [self.tok_to_id(token, train=train) for token in tok_seq]

    def tok_to_id(self, token, train=False):
        """ Maps a token to it's corresponding ID. Or if in training mode, also adds new words to the vocab
        Args:
            token: A word
        Returns:
            An ID
        """

        if train is True:
            self.token_counter += 1
            return self.vocab[token]            
        elif token in self.vocab:
            return self.vocab[token]
        else:
            return self.vocab[UNK]

    def make_reverse_vocab(self):
        """ Makes a reverse vocab for the given vocab.
        """
        self.reverse_vocab = {id_:token for token,id_ in self.vocab.items()}

    def ids_to_text(self, id_list):
        """ Maps a sequence of IDs to a string
        Args:
            id_list: A sequence of IDs
        Returns:
            text: A text string that is supposed to be a sentencentences: the list of split sentences.
        """

        tokens = ''.join(map(lambda x:self.reverse_vocab[x],id_list))

        return tokens

    def ids_to_string(self,tokens,length=None):
        """ Converts a text of ids(kind of like the output of the model) into a readable format with words.
        Args:
            tokens: A string comprised of tokens
            length: The length of the string you want it to tokenize
        Returns:
            string: The corresponding string for the ids with words in them
        """
        
        string = ''.join([self.reverse_vocab[x] for x in tokens[:length]])
        
        return string



    def create_embedding_matrix(self, path=None):
        """ Reads the FastText word embeddings from a file, adds new embeddings to vocab, and fills in any word embeddings
        Args:
            path: path to the .vec file
        Returns:
            embeddings: A numpy array of vectors
        """

        if path is None:
            raise Exception('You must specify a path to the pretrained embeddings...')

        fin = io.open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
        length, dim = map(int, fin.readline().split())
        
        print(format("Reading word embedding file with length %d and dim %d", length, dim))

        data = {}
        for line in fin:
            tokens = line.rstrip().split(' ')
            data[tokens[0]] = map(float, tokens[1:])
            self.vocab[tokens[0]] # add the token to the vocab


        print("Adding embeddings to the words in the vocab...")

        # TODO: Handle Special tokens 
        embeddings = list()
        count = 0
        for word, id_ in self.vocab:
            if word in [PAD, UNK, START, EOS]:
                continue

            if word in data.keys():
                # import the vector from the data dictionary
                embeddings.append(data[word])
            else:
                count += 1
                embeddings.append(self.generate_embedding(word))

        print("There were %d words without pretrained vectors" % count)

        # Convert to numpy array
        embeddings = np.array([np.array(i) for i in embeddings])

        return embeddings

    def generate_embedding(self, word):
        """ Generates an embedding for an OOV word with FastText Pretrained model
        Args:
            word: The word to generate an embedding for
        Returns:
            vector: An 300 dimensional vector is returned
        """

        # Load the fattext format
        model = ft.load_fasttext_format(WORD_VECS)

        result = model.similar_by_word(word=word, topn=2)

        vector = np.mean(np.array([model[result[0]],model[result[1]]]), axis=0)

        vector = vector.tolist()

        return vector

    def save_vocab(self, path=None):
        """ Saves the vocabulary to file.
        """

        if path is None:
            print("No path for vocab specified. Saving to:", os.getcwd())

        return NotImplementedError

    def load_vocab(self, path=None):
        """ Loads the vocab file from the specified path
        """

        if path is None:
            raise Exception("Error: A path must be specified!")

        return NotImplementedError

