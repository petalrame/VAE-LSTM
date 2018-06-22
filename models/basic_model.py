""" The base class for any given model. Contains the basic requirements common to most models. """

import os, copy
import json
import tensorflow as tf

class BasicAgent(object):

    def __init__(self, config):
        # load config for hyperparams
        if config['force']:
            config.update(self.load_config(config['config_path']))

        # save config
        if config['save_config']:
            self.save_config(config['config_path'])

        # keep a copy of the configuration to avoid mutation of the configuration
        self.config = copy.deepcopy(config)

        # print configuration if in debug mode
        if config['debug']:
            print('config', self.config)

        # the child model MUST have its own build graph function
        self.graph = self.build_graph(tf.graph())

        # add in any operations that are common to all model
        with self.graph.as_default():
            self.saver = tf.train.Saver(max_to_keep=3)

        # add some other initialization code here
        gpu_options = tf.GPUOptions(allow_growth=True, allow_soft_placement=True)
        sessConfig = tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.Session(config=sessConfig, graph=self.graph)
        self.sw = tf.summary.FileWriter(self.result_dir, self.sess.graph)

        # this function is not always common to all models as some models need further initialization and other do not
        # this is also why this function is separate from the __init__() function
        self.init()
        
        # The model should now be ready!

    def set_agent_props(self):
        """ Sets the custom parameters for the model.
        """

        print("WARNING: This method should be completely overwritten.")

        pass

    def load_config(self, path):
        """ Loads a configuration at a specified path
        """

        # check to see if the configuration exists
        if path is None:
            raise Exception('You must specify a path to the configuration')

        try:
            with open(path, 'r') as fname:
                config = json.loads(fname)
        except:
            raise Exception('There was an error trying to load this config. Check that config exists')

        return config

    def save_config(self, path):
        """ Save current config
        Args:
            path: Path with filename to save config
        """

        if path is None:
            raise Exception('ERROR: config_path is not set')

        if os.path.isdir(path):
            raise Exception('ERROR: config_path should be a path to a file not an existing directory')

        # dump config to JSON
        try:
            with open(path, 'w') as fname:
                json.dump(self.config, fname)
        except:
            raise Exception('ERROR: There was a problem dumping the config to file')

    @staticmethod
    def get_random_config(fixed_params={}):
        """ Generates a completely random configuration of the model.
        Properties:
            @staticmethod: So we canm create an independant random process for each model
        """

        raise Exception("The get_random_config function must be overriden by the agent")

    def build_graph(self, graph):
        """ Builds the computational graph
        Args:
            graph: A(assumed blank) computational graph 
        Returns:
            Built out computational graph
        """

        raise Exception("The build_graph function must be overriden by the agent")

    def infer(self):
        """ Builds and runs one instance of the graph for inference
        """

        raise Exception("The infer function must be overriden by the agent")

    def train_step(self):
        """ Trains a model for one step/batch
        """

        raise Exception("The learn_from_epoch function must be overriden by the agent")

    def train(self, save_every=1):
        """ Defines the training loop for the model.
        Args:
            save_every: Epoch increments to save in. Pass negative number for no saving
        """
        for epoch_id in range(0, self.max_iter):
            self.learn_from_epoch()

    def eval_step(self):
        """ Runs the model for one step/batch
        """

        raise Exception('The eval_step function must be overriden by the agent')

    def eval(self):
        """ Runs the eval loop until terminated
        """

        raise Exception('The eval function must be overriden by the agent')

    def init(self):
        """ Initializes some global stuff for the model. Should be common to most models.
        Making it separate from the __init__ allows us to override this cleanly for all models
        """

        # This is an example of a useful init function
        checkpoint = tf.train.get_checkpoint_state(self.result_dir)
        if checkpoint is None:
            self.sess.run(self.init_op)
        else:
            if self.config['debug']:
                print('Loading the model from folder: %s' % self.result_dir)
            self.saver.restore(self.sess, checkpoint.model_cehckpoint_path)

    


