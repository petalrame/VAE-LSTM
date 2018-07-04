# Data Pipeline

Building an input pipeline can take forever, so here we take advantage of some high level APIs that Tensorflow gives us to make a smooth and efficient input pipeline.

## Overview of Requirements
1. Downloading Data
2. Splitting Data
3. Cleaning Data
4. Building Vocabulary
5. Serializing Data

## Downloading Data
This section will involve code found in ```src/data/download.py```

Downloading data is going to specific what what you want to download. However, steps can be taken to create general download functionality.
Another thing that we will have to deal with is the automatic cleanup and extraction of the data files which can be handled automatically as well
, but the cleanup/extraction code will be specific to the project at hand

The function ```download_file(url, data_dir)``` handles the downloading of the dataset or other useful files like pretrained word vectors.
The function ```zip_handler(zipf, path)``` handles extracting certain data files(this function is not as general and is dependant on the data it's meant to extract)

The last function could be made more general by including a search string of files to keep and name of directory to search for the data to keep.

## Splitting Data
This section will involve code found in ```src/data/download.py```

Splitting the data is a choice that needs to be made on the part of the user running the experiment and should be baked in.
An arbitrary split can be used and is left up the user.

In this project, the MSCOCO dataset had already been split into train/val. We just used the same proportion from MSCOCO to split the
QUORA dataset.

## Cleaning Data
This section will involve code found in ```src/data/download.py``` and ```src/data/vocab.py```

Cleaning the data is an important preprocessing tool. It involves removing troubling characters, converting a sentence to lowercase and splitting it into tokens.

All this is acheived in the ```src/data/vocab.py``` file where the Vocab object is. The NLTK word_tokenizer is used to convert a cleaned text into tokens.
The ```_tokenize()``` function in vocab.py takes care of all of that. 

## Building Vocabulary
This section will involve code found in ```src/data/vocab.py```

Building the vocab involves converting a list of tokens to their corresponding IDs **FROM THE TRAIN DATA** and adding said mapping to some dictionary.
You can see the vocab.py file for all the functions available to map, add, save and load the vocabulary.

The function ```prep_train_seq``` adds unseen tokens to the vocab. This function does not map anything because it's only a pass over the train data to build the vocab.
The vocab built on the train data can additionally be limited to the most frequent words on save through the ```save_vocab(path, max_keep=None)``` function. Setting max_keep to None will make save all the words seen in the train set.

After the vocabulary is built and saved to file, the text in the dataset must be mapped to IDs. The function ```prep_seq(seq)``` is used for this.

## Serializing Data
This section will involve code found in ```src/data/handler.py``` and specifically in the ```_make_example(sequence, target)``` function

After the vocabulary is built and an example is converted from raw text to a list of IDs, it is time for the data to be serialized to tfrecord format(the reccomended file format to use with training Tensorflow models). The serialization uses protobuf files to create an instance of a tf.train.SequenceExample. 

The tf.train.SequenceExample is composed of two parts: context & feature_lists[**NOTE**: The exact spelling is super important]
1. The **context** is for scalar quantities we want to feed in. It is composed of a python dict mapping the following: name : tf.train.feature that is composed of a Int64List. In our case, we feed in the sequence and target lengths used for the dynamic RNN so that it doesn't compute extra steps on the padded inputs.

2. The **feature_lists** is for sequential features. It takes a dict mapping the following: name : feature_list that is composed of Int64List features for each ID.

After the SequenceExample is returned, the example is added to the tfrecord being written with ```ex.SerializeToString()``` method

Every project will have a similar version of the this function. The only change will be in the names and the data types involved(as a sequence2sequence model might not always be the one needing the preprocessing pipeline). For cases outside of the sequence2sequence case, the below resources might prove extremely useful.

## Testing
This section will involve the code found in ```src/tests/```

**IN PROGRESS**

## Resources
1. Stanford guide to building an input pipeline(useful for NLP) => ["LINK"](https://cs230-stanford.github.io/tensorflow-input-data.html)
2. A guide on the tf.Data and Estimator APIs => ["LINK"](http://ruder.io/text-classification-tensorflow-estimators/)
3. An example of how tfrecords can be serialized and deserialized => ["LINK"](https://github.com/yxtay/text-classification-tensorflow/blob/master/acl_imdb.py)