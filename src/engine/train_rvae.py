""" This is the command line interface for the implementation of the Recurrent Variational Autoencoder """
from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf

from src.data.dataset import Dataset
from src.data.vocab import Vocab

FLAGS = tf.app.flags.FLAGS

# Where to find data
tf.app.flags.DEFINE_string('data_path', '', 'Path to the tf.Record data files.')
tf.app.flags.DEFINE_string('vocab_path', '', 'Path to the saved vocab file.')
tf.app.flags.DEFINE_string('embed_path', '', 'Path to the serialized pretrained embedding matrix.')

# Model settings
tf.app.flags.DEFINE_string('mode', 'train', 'must be one of train/eval/predict')

# Where to save the outputs of your experiments
tf.app.flags.DEFINE_string('model_dir', '', 'Directory to store all the outputs(logs/checkpoints/etc). Must be provided for eval and predict mode(s)')
tf.app.flags.DEFINE_string('exp_name', '', 'Name of the experiment. Results dir will have this name')

# Hyperparameters
tf.app.flags.DEFINE_integer('batch_size', 32, 'size of the mini batches of data')
tf.app.flags.DEFINE_integer('emb_dim', 300, 'dimension of the word embeddings')
tf.app.flags.DEFINE_integer('hidden_dim', 600, 'size of the RNN hidden states')
tf.app.flags.DEFINE_integer('latent_dim', 1100, 'size of the latent space')
tf.app.flags.DEFINE_integer('dec_layers', 2, 'number of layers for the decoder')
tf.app.flags.DEFINE_integer('beam_size', 10, 'beam size for beam search decoding')
tf.app.flags.DEFINE_integer('max_dec_steps', 200, 'max time steps allowed for decoding')
tf.app.flags.DEFINE_integer('vocab_size', 50000, 'size of vocab')
tf.app.flags.DEFINE_float('lr', 0.00005, 'the learning rate')
tf.app.flags.DEFINE_float('keep_prob', 0.7, '1 - dropout rate')

# Debugging
tf.app.flags.DEFINE_boolean('debug', False, "Run in tensorflow's debug mode")

def main():
    #TODO: Check flags and do stuff
    raise NotImplementedError

if __name__ == '__main__':
    tf.app.run()