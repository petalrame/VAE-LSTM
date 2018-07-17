""" Implementation of the Reccurrent Variational Autoencoder """

import tensorboard
# Globals kept here
import tensorflow as tf
from tensorflow import layers


class RVAE(object):
    """ Builds the model graph for different modes(train, eval, predict) """
    def __init__(self, hps):
        self.hps = hps

    def embedding(self, input):
        """ Create the embedding layer to be used for taking inputs from the input_fn """
        vsize = self.hps['vocab_size']
        emb_dim = self.hps['embedding_dimension']
        with tf.variable_scope('embedding'):
            embedding = tf.get_variable('embedding', [vsize, emb_dim], dtype=tf.float32)
        return NotImplementedError

    def encoder(self, input, mode):
        """ Creates the encoding layer for the RVAE """
        return NotImplementedError

    def vae(self, input):
        """ Builds and creates the variational autoencoder ops """
        return NotImplementedError

    def decoder(self, input, mode):
        """ Creates the decoding layer for the RVAE """
        return NotImplementedError

    def train_op(self):
        """ Creates and adds training ops to the graph """
        return NotImplementedError

    def model_fn(self, features, labels, mode, params):
        """ Builds the graph of the model being implemented 
        Args:
            features: A Tensor or dict of Tensors to be used as features(input)
            labels: A Tensor or doct of Tensors to be used as labels. Should be blank for
            predict mode.
            mode: An instance of tf.estimator.ModeKeys to be used for calls to train() and evaluate()
            params: Any additional configuration needed    
        Returns:
            tf.estimator.EstimatorSpec which is contains information the caller(i.e train(), evaluate(), predict())
            needs.
        """
        return NotImplementedError

