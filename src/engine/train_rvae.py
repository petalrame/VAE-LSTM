""" This is the command line interface for the implementation of the Recurrent Variational Autoencoder """
from __future__ import absolute_import, division, print_function

import os
import sys

import numpy as np
import tensorflow as tf

from collections import namedtuple
sys.path.append('../')
from src.data.dataset import Dataset
from src.data.vocab import Vocab
from src.models.rvae import RVAE
from src.utils.load_config import ModelParams, AppConfig

FLAGS = tf.app.flags.FLAGS

# Use config files insetad of tensorflow app flags
tf.app.flags.DEFINE_string('app_config', ''. 'profile for the app configuration')
tf.app.flags.DEFINE_string('model_params', ''. 'profile for the model hyperparameters')

# Where to find data
tf.app.flags.DEFINE_string('data_path', '/home/tldr/Projects/models/current/VAE-LSTM/data/processed/train.tfrecord', 'Path to the tf.Record data files or text file if predicting.')
tf.app.flags.DEFINE_string('eval_path', '/home/tldr/Projects/models/current/VAE-LSTM/data/processed/val.tfrecord', 'Path to the tf.Record data file for eval.')
tf.app.flags.DEFINE_string('vocab_path', '/home/tldr/Projects/models/current/VAE-LSTM/data/processed/vocab.tsv', 'Path to the saved vocab file.')
tf.app.flags.DEFINE_string('embed_path', '/home/tldr/Projects/models/current/VAE-LSTM/data/external/crawl-300d-2M.vec', 'Path to the serialized pretrained embedding matrix.')
tf.app.flags.DEFINE_string('checkpoint_path', '', 'Path to the model checkpoint.')

# Model settings
tf.app.flags.DEFINE_string('mode', 'train', 'must be one of train/predict/save_embed/eval')

# Where to save the outputs of your experiments
tf.app.flags.DEFINE_string('model_dir', '/home/tldr/Projects/models/current/VAE-LSTM/results/', 'Directory to store all the outputs(logs/checkpoints/etc). Must be provided for eval and predict mode(s)')
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

def infer(model, ds, vocab, checkpoint_path=None):
    """ Runs a saved model in inference mode
    """
    # get the embedding matrix
    emb_init = vocab.read_embeddings(FLAGS.embed_path)

    # get config
    if FLAGS.debug:
        sess_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    else:
        sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True #pylint: disable=E1101
    sess_config.gpu_options.per_process_gpu_memory_fraction = 0.9 #pylint: disable=E1101
    config = tf.estimator.RunConfig(model_dir=FLAGS.model_dir,
                                    save_summary_steps=100,
                                    session_config=sess_config)

    # make estimator
    estimator = tf.estimator.Estimator(
        model_fn=model.model_fn,
        model_dir=FLAGS.model_dir,
        config=config,
        params={'embedding_initializer': emb_init})


    input_fn = ds.predict_input_fn(path=FLAGS.data_path, batch_size=FLAGS.batch_size)
    predictions = estimator.predict(input_fn=input_fn, checkpoint_path=checkpoint_path)

    vocab.make_reverse_vocab()

    for ex in predictions:
        print(vocab.ids_to_text(ex['pred']))
    
    return list(predictions)

def eval(model, ds, vocab):
    """ Runs the eval loop
    """
    # get the embedding matrix
    emb_init = vocab.read_embeddings(FLAGS.embed_path)

    # get config
    if FLAGS.debug:
        sess_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    else:
        sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True #pylint: disable=E1101
    sess_config.gpu_options.per_process_gpu_memory_fraction = 0.9 #pylint: disable=E1101
    config = tf.estimator.RunConfig(model_dir=FLAGS.model_dir,
                                    save_summary_steps=100,
                                    session_config=sess_config)

    # make estimator
    estimator = tf.estimator.Estimator(
        model_fn=model.model_fn,
        model_dir=FLAGS.model_dir,
        config=config,
        params={'embedding_initializer': emb_init})

    result = estimator.evaluate(input_fn=lambda:ds.train_input_fn(FLAGS.eval_path, FLAGS.batch_size))

    print(result)


def train_and_eval(model, ds, vocab):
    """ Runs train and eval simultaneously
    """
    # get the embedding matrix
    emb_init = vocab.read_embeddings(FLAGS.embed_path)

    # TODO: enable mirrored distribution strategy
    #distribute = tf.contrib.distribute.MirroredStrategy()

    # get config
    if FLAGS.debug:
        sess_config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
    else:
        sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True #pylint: disable=E1101
    sess_config.gpu_options.per_process_gpu_memory_fraction = 0.9 #pylint: disable=E1101
    config = tf.estimator.RunConfig(model_dir=FLAGS.model_dir,
                                    save_summary_steps=100,
                                    save_checkpoints_steps=100,
                                    session_config=sess_config)

    # make estimator
    estimator = tf.estimator.Estimator(
        model_fn=model.model_fn,
        model_dir=FLAGS.model_dir,
        config=config,
        params={'embedding_initializer': emb_init}
    )

    # make the BestExporter
    spec = {
        "source_len": tf.FixedLenFeature([], dtype=tf.int64),
        "source_seq": tf.VarLenFeature(dtype=tf.int64)
    }
    serving_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec=spec)
    exporter = tf.estimator.BestExporter(name='best_exporter', serving_input_receiver_fn=serving_input_fn, exports_to_keep=5)

    # call the train_and_evaluate method
    train_spec = tf.estimator.TrainSpec(input_fn=lambda:ds.train_input_fn(FLAGS.data_path, FLAGS.batch_size), max_steps=FLAGS.train_iterations)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda:ds.train_input_fn(FLAGS.eval_path, FLAGS.batch_size), exporters=exporter, start_delay_secs=0, throttle_secs=0)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

def example_serving_input_fn():
    """ The serving input function.
    """
    # add features
    ex_context = {
        "source_len": tf.FixedLenFeature([], dtype=tf.int64)
    }
    ex_sequence = {
        "source_seq": tf.FixedLenSequenceFeature([], dtype=tf.int64)
    }

    # placeholder for tf.Example
    serialized_ex = tf.placeholder(
        shape=[],
        dtype=tf.string,
        name="input_tensors"
    )

    # parse features
    context, sequence = tf.parse_single_sequence_example(
        serialized=serialized_ex,
        context_features=ex_context,
        sequence_features=ex_sequence)

    receiver = {"inputs": serialized_ex}
    features = {"source_seq": sequence["source_seq"], "source_len": context["source_len"]}
    
    return tf.estimator.export.ServingInputReceiver(features, receiver)

def main(unused_argv):
    if len(unused_argv) != 1:
        raise Exception('Problem with number of flags entered %s' % unused_argv)

    tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.info('Starting RVAE model in %s mode', (FLAGS.mode))

    if FLAGS.mode == 'train':
        assert FLAGS.eval_path is not None, "Error! Eval path must be provided in 'train' mode. Use train_only for only training"
    elif FLAGS.mode == 'predict':
        FLAGS.batch_size=1

    # change model_dir to model_dir/exp_name and create dir if needed
    FLAGS.model_dir = os.path.join(FLAGS.model_dir, FLAGS.exp_name)
    if not os.path.exists(FLAGS.model_dir):
        if FLAGS.mode == 'train' or FLAGS.mode == 'save_embed':
            os.makedirs(FLAGS.model_dir)
        else:
            raise Exception("The model_dir specified does not exist. Run in train to create it")
    elif not os.path.exists(FLAGS.vocab_path):
        raise Exception("Path specified for vocab file does not exist")
    elif FLAGS.checkpoint_path and not os.path.exists(FLAGS.checkpoint_path):
        raise Exception("Path for checkpoint does not exist")

    # load vocab and calculate size
    vocab = Vocab(vocab_path=FLAGS.vocab_path)
    vsize = len(vocab.vocab)

    # load the dataset
    ds = Dataset(vocab)

    # create an hps list
    if not FLAGS.app_config or not FLAGS.model_params:
        hp_list = ['batch_size', 'emb_dim', 'hidden_dim', 'latent_dim', 'dec_layers', 'beam_size', 'max_dec_steps', 'lr', 'keep_prob', 'use_wdrop', 'model_dir', 'vocab_path']
        hps_dict = {}
        for key in FLAGS:
            if key in hp_list:
                hps_dict[key] = FLAGS[key].value
        hps = namedtuple("HParams", hps_dict.keys())(**hps_dict)
    else:
        # TODO: Fix the yaml file location to use relative pathing so that we can find the config files
        FLAGS = AppConfig('app.yaml', 'default')
        hps = ModelParams('hps.yaml', 'default')

    # call the model
    model = RVAE(hps, vsize)

    if FLAGS.mode == 'train':
        train_and_eval(model, ds, vocab)
    elif FLAGS.mode == 'predict':
        predictions = infer(model, ds, vocab, FLAGS.checkpoint_path)
        print(predictions)
    elif FLAGS.mode == 'eval':
        eval(model, ds, vocab)
    elif FLAGS.mode == 'save_embed':
        _ = vocab.read_embeddings(path=FLAGS.embed_path, load_np=False)
        print("Done saving numpy matrix")
        return
    elif FLAGS.mode == 'debug':
        print("debug")
    else:
        raise Exception("Invalid mode argument")

    return

if __name__ == '__main__':
    tf.app.run()
