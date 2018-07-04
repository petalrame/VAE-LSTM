# VAE-LSTM

Implementation of "A Deep Generative Framework for Paraphrase Generation"(**VAE-SVG ONLY**)

## File Structure

```
├── README.md
├── data
│   ├── external => External Data(e.g word vectors)
│   ├── interim => Cleaned data before undergoing writing to tfrecord
│   ├── processed => tfrecord files for train/eval and vocab
│   └── raw => Downloaded or collected data
├── docs => Contain some notes on the implementation 
├── results => Output of experiments
└── src
    ├── data => Preprocessing code(i.e downloading, cleaning data and making vocab)
    ├── external => I'm thinking this is where the CLIs are gonna go
    ├── models => Contains any and all models created
    ├── tests => Contains tests for preprocessing and the models
    ├── tools => Contains utilities and eval scripts(e.g BLEU script for NMT)
    └── visualization => Any sort of code used for making visualizations of the data
```

For more on file structure see [3]

## Installation

I will describe all the necessary dependencies required for running this application here. **IN PROGRESS**

## Usage

I will describe all the commands that you can use to train and evaluate the model here. **IN PROGRESS**

## Test

I will describe the commands for testing the model here. **IN PROGRESS - I HAVE YET TO MAKE TESTS**

## Useful Links

1. The Paper - ["A Deep Generative Framework for Paraphrase Generation"](https://www.cse.iitk.ac.in/users/piyush/papers/deep-paraphrase-aaai2018.pdf)
2. Blogpost on Dataset AP(Better ways to do this found in resources in the docs) - ["Getting Text into Tensorflow with the Dataset API"](https://medium.com/@TalPerry/getting-text-into-tensorflow-with-the-dataset-api-ffb832c8bec6)