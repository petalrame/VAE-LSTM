""" This is the command line interface for the implementation of the Recurrent Variational Autoencoder """
from __future__ import absolute_import, division, print_function

import os

import numpy as np
import tensorflow as tf

from collections import namedtuple
from src.data.dataset import Dataset
from src.data.vocab import Vocab
from src.models.rvae import RVAE

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
tf.app.flags.DEFINE_integer('train_iterations', 600000,'the number of training iterations to perform')
tf.app.flags.DEFINE_integer('batch_size', 32, 'size of the mini batches of data')
tf.app.flags.DEFINE_integer('emb_dim', 300, 'dimension of the word embeddings')
tf.app.flags.DEFINE_integer('hidden_dim', 600, 'size of the RNN hidden states')
tf.app.flags.DEFINE_integer('latent_dim', 1100, 'size of the latent space')
tf.app.flags.DEFINE_integer('dec_layers', 2, 'number of layers for the decoder')
tf.app.flags.DEFINE_integer('beam_size', 10, 'beam size for beam search decoding')
tf.app.flags.DEFINE_integer('max_dec_steps', 200, 'max time steps allowed for decoding')
tf.app.flags.DEFINE_float('lr', 0.00005, 'the learning rate')
tf.app.flags.DEFINE_float('keep_prob', 0.7, '1 - dropout rate')
tf.app.flags.DEFINE_boolean('use_wdrop', True, 'Use word dropout as described in arxiv 1511.06349')

# Debugging
tf.app.flags.DEFINE_boolean('debug', False, "Run in tensorflow's debug mode")

def train(model, ds):
    """ Trains the model only
    """
    return NotImplementedError

def eval(model, ds):
    """ Evaluates the model only
    """
    return NotImplementedError

def infer(model, ds):
    """ Runs a saved model in inference mode
    """
    return NotImplementedError

def train_and_eval(model, ds):
    """ Runs train and eval simultaneously
    """
    return NotImplementedError

def main(unused_argv):
    #TODO: Check flags and do stuff
    if len(unused_argv) != 1:
        raise Exception('Problem with number of flags entered %s' % unused_argv)

    tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.info('Starting RVAE model in %s mode', (FLAGS.mode))

    # change model_dir to model_dir/exp_name and create dir if needed
    FLAGS.model_dir = os.path.join(FLAGS.model_dir, FLAGS.exp_name)
    if not os.path.exists(FLAGS.model_dir):
        if FLAGS.mode == 'train':
            os.makedirs(FLAGS.model_dir)
        else:
            raise Exception("The model_dir specified does not exist. Run in train to create it")
    elif not os.path.exists(FLAGS.vocab_path):
        raise Exception("Path specified for vocab file does not exist")

    # load vocab and calculate size
    vocab = Vocab(vocab_path=FLAGS.vocab_path)
    vsize = len(vocab.vocab)

    # load the dataset
    ds = Dataset(vocab)

    # create an hps list
    hp_list = ['batch_size', 'emb_dim', 'hidden_dim', 'latent_dim', 'dec_layers', 'beam_size', 'max_dec_steps', 'lr', 'keep_prob', 'use_wdrop', 'model_dir', 'vocab_path']
    hps_dict = {}
    for key, val in FLAGS.__flags.items():
        if key in hp_list:
            hps_dict[key] = val
    hps = namedtuple("HParams", hps_dict.keys())(**hps_dict)

    # call the model
    model = RVAE(hps, vsize)

    if FLAGS.mode == 'train':
        train_and_eval(model, ds)
    elif FLAGS.mode == 'eval':
        eval(model, ds)
    elif FLAGS.mode == 'infer':
        infer(model, ds)
    elif FLAGS.mode == 'train_only':
        train(model, ds)
    else:
        raise Exception("Invalid mode argument")

    return

if __name__ == '__main__':
    tf.app.run()
