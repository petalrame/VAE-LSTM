""" Implementation of the Reccurrent Variational Autoencoder """
from __future__ import absolute_import, division, print_function

import tensorboard
import tensorflow as tf
from tensorflow import keras, layers


class RVAE(object):
    """ Builds the model graph for different modes(train, eval, predict) """
    def __init__(self, hps):
        self.hps = hps

    def embedding(self, inputs, emb_tensor, emb_dim):
        """ Create the embedding layer that parses the input tensor 
        Args:
            inputs: An iterable of `Tensor`s
            emb_tensor: A `Tensor` that contains the embedding for each token in vocab
            emb_dim: `int`, number of embedding dimensions
        Returns:
            An iterable of tensors corresponding to [batch_size, seq_len, emb_dim]
        """
        return NotImplementedError

    def encoder(self, input, hidden_dim, scope, reuse):
        """ Creates an encoder for parsing inputs
        Args:
            input: `Tensor`, input tensor
            hidden_dim: `int`, size of the hidden dimension for the LSTMCell
            initializers: specify/pass initializers for variables
            scope: `string`, allows variable reuse or creation of new ones with same function
            reuse: `bool`, indicates if variable(s) should be reused if it's present in scope
        Returns:
            fwd_state, bw_state: Forward and backward states of the encoder
        """
        return NotImplementedError

    def vae(self, input):
        """ Builds and creates the variational autoencoder ops """
        return NotImplementedError

    def decoder(self, input, mode):
        """ Creates a decoder to produce outputs """
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
