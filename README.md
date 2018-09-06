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
```
Requirements:
    - Python: 3.6.5
    - Numpy: Latest
    - tensorflow-gpu: r1.10 (compiled from master branch due to bug in the pip install version)
    - NLTK: Latest
    - tqdm: Latest
    - pytest: Latest
```

## Usage

I will describe all the commands that you can use to train and evaluate the model here. **TODO**
### Downloading the data
The downloading and preprocessing of data happens in the ```src/data/download.py``` script.
Once it has been downloaded and parsed into tfrecord files, you should create the numpy embedding matrix.

### Persist Embedding Tensor
To do this, you first have to be in the ```src/``` directory.
Once there, run ```python engine/train_rvae.py --mode=save_embed```
**NOTE ON DATA PATHS**: There are some hard coded flags in there because I'm lazy and didn't make a config file to find the location of my data. Just
change the paths at the top of the "train_rvae.py" file.

### Train and Evaluate the Model
Once you have the persisted embedding tensor, run the following command to train and evaluate the model:
```python engine/train_rvae.py --mode=train --model_dir=[DIRECTORY TO SAVE THE MODEL TO] --exp_name=[NAME OF EXPERIMENT] --data_path=[PATH TO DATA]```

### Run Tensorboard
```tensorboard --logdir={model_dir}```

### Run the Model in Predict Mode
The PREDICT mode reads a .txt file and feeds in subsequent sentences for paraphrase generation.
To run the trained model in prediction mode, run the following command
```python engine/train_rvae.py --mode=predict --model_dir=[PATH TO MODEL DIR] --data_path=[PATH TO TXT FILE]```

## Test
More tests(and better) tests will be implemented in the future. For now it only tests to see if the input_fn spits out the correct output.

## Utils
I've included a small script to inspect checkpoint files for the model. All you have to do is replace the hardcoded line in the script for the
model path and run it.

## Useful Links

1. The Paper - ["A Deep Generative Framework for Paraphrase Generation"](https://www.cse.iitk.ac.in/users/piyush/papers/deep-paraphrase-aaai2018.pdf)
2. Blogpost on Dataset API(Better ways to do this found in resources in the docs) - ["Getting Text into Tensorflow with the Dataset API"](https://medium.com/@TalPerry/getting-text-into-tensorflow-with-the-dataset-api-ffb832c8bec6)