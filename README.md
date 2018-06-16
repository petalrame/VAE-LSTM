# VAE-LSTM
Implementation of "A Deep Generative Framework for Paraphrase Generation"(**VAE-SVG ONLY**)

## File Structure
```
|- data/
| |- __init__.py
| |- download.py --> Downloads dataset and splits it into train/test/eval
| |- handler.py --> Makes TFRecords from the data
| |- vocab.py --> Handles the vocab
|- models/
| |- __init__.py
| |- basic_model.py --> Base class for all models
| |- model.py --> Class for our implementation
|- results/
| |- .gitkeep
|- tests/
| |- [VARIOUS TEST FILES]
|- utils/
| |- [HPSEARCH FILES]
| |- [MISC. FILES] --> e.g saving/loading etc.
|- main.py --> Training, inference, eval, TF app entry
|- .gitignore
|- README.md
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
2. Blogpost on Dataset API - ["Getting Text into Tensorflow with the Dataset API"](https://medium.com/@TalPerry/getting-text-into-tensorflow-with-the-dataset-api-ffb832c8bec6)
3. Blogpost on Good Practices - ["Tensorflow: A proposal of good practices for files, folders and models architecture"](https://blog.metaflow.fr/tensorflow-a-proposal-of-good-practices-for-files-folders-and-models-architecture-f23171501ae3)