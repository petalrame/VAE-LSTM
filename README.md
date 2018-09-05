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
    ├── engine => Entry to train/eval/predict the rvae model
    ├── models => Contains any and all models created(specifically RVAE here and more variations of the architecture in the future)
    ├── tests => Contains tests for preprocessing and the models(right now I only have some for the data package and they need to be way better)
    └── tools => Contains utilities and eval scripts(like BLEU script for NMT)
```


## Installation

I will describe all the necessary dependencies required for running this application here. **TODO**

## Usage

I will describe all the commands that you can use to train and evaluate the model here. **TODO**

## Test

I will describe the commands for testing the model here. **IN PROGRESS - MORE TESTS NEEDED**

## Useful Links

1. The Paper - ["A Deep Generative Framework for Paraphrase Generation"](https://www.cse.iitk.ac.in/users/piyush/papers/deep-paraphrase-aaai2018.pdf)
2. Blogpost on Dataset API(Better ways to do this found in resources in the docs) - ["Getting Text into Tensorflow with the Dataset API"](https://medium.com/@TalPerry/getting-text-into-tensorflow-with-the-dataset-api-ffb832c8bec6)