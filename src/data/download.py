""" Download data and pretrained vectors and split into Train/Val """
import csv
import glob
import json
import math
import os
import re
import shutil
import sys
import zipfile
from collections import defaultdict
from random import shuffle

import requests
from tqdm import tqdm

from vocab import Vocab
from handler import Dataset

# put your data directories here
WORKING_DIR = os.path.abspath(os.path.dirname(__file__)) # path to file
BASE_DIR = os.path.abspath(os.path.join(WORKING_DIR, "../../data"))
RAW_DIR = os.path.join(BASE_DIR, "raw")
EXTERNAL_DIR = os.path.join(BASE_DIR, "external") # holds external data(pretrained embeddings)
INTERIM_DIR = os.path.join(BASE_DIR, "interim") # hold data before tf processing is done
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")

# Put dataset URLs and filenames here
QUORA = "http://qim.ec.quoracdn.net/quora_duplicate_questions.tsv"
MSCOCO = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
FASTTEXT = "https://s3-us-west-1.amazonaws.com/fasttext-vectors/crawl-300d-2M.vec.zip"
COCO_TRAIN = os.path.join(RAW_DIR,"captions_train2014.json") 
COCO_VAL = os.path.join(RAW_DIR,"captions_val2014.json")
QUORA_RAW = os.path.join(RAW_DIR,"quora_duplicate_questions.tsv")

# Name of intermediary files
TRAIN_DATA = os.path.join(INTERIM_DIR, "train_data.csv")
VAL_DATA = os.path.join(INTERIM_DIR, "val_data.csv")
TRAIN_RECORD = os.path.join(PROCESSED_DIR, "train.tfrecord")
VAL_RECORD = os.path.join(PROCESSED_DIR, "val.tfrecord")

def download_file(url, data_dir):
    """ General purpose function to download a file to working dir.
    Args:
        url: URL of the resource to download
        data_dir: Directory to save the file to
    Returns:
        file_path: The path where the file was downloaded
    """
    
    # No existing file, let's download this file
    print("Downloading file located at:", url)

    if not os.path.isdir(data_dir):
        raise Exception("Path was None or is not a directory")
    else:
        file_path = data_dir
        print("Saving file to:", data_dir)
    
    # Make request for file and get content length for tqdm
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    wrote = 0

    # Find name of the file
    if url.find('/'):
        fname = url.rsplit('/', 1)[1]
        file_path = os.path.join(file_path, fname)

    # Check to see if we have already downloaded to this location
    if os.path.exists(file_path):
        print(file_path, "already exists. Skipping...")
        return file_path

    # Show progress of download and write to file
    with open(file_path, 'wb') as f:
        for chunk in tqdm(response.iter_content(chunk_size=1024), total=total_size/1024, unit='KB'):
            if chunk:
                wrote += len(chunk)
                f.write(chunk)

    # Some error handling
    if total_size != 0 and wrote != total_size:
        raise ValueError('Did not download the full file. Something went wrong')

    return file_path

def zip_handler(zipf, path):
    """ Handle the unzipping, moving and deleting of a specific dataset.
    NOTE: This function heavily depends on the dataset being handled and should be changed from project to project!
    Args:
        zipf: A zip file to unpack and process
        path: Path to extract to
    """

    # Unzip the file
    with zipfile.ZipFile(zipf, "r") as zip_ref:
        zip_ref.extractall(path)

    # For the MSCOCO dataset(not needed for Quora)
    del_dir = os.path.join(path, "annotations")

    # Find files to keep
    keep_files = glob.glob(os.path.join(del_dir,'captions_*.json'))
    print("Keeping the following files:", keep_files)

    # Move the files to keep into the working dir
    for file in keep_files:
        shutil.move(file, path)

    # Cleanup directory and zip file
    shutil.rmtree(del_dir)
    os.remove(zipf)

    return

def handle_coco(vocab):
    """ Handles all parsing of the raw MSCOCO dataset.
    This includes getting keeping only 4 captions per image and writing the dataset into CSV
    Args:
        vocab: vocab object to build vocab with
    Returns:
        idx/total: Split from validation to train data
    """
    total = 0
    split = 0.0

    print("Processing MSCOCO dataset...")
    for dataset in [COCO_VAL, COCO_TRAIN]:
        temp_dict = defaultdict(list) # we store the data here for refinement
        # Read and parse the JSON
        with open(dataset) as coco_file:
            data = json.load(coco_file)
            for _, anno in enumerate(data["annotations"]):
                sent = anno["caption"].rstrip()
                temp_dict[anno["image_id"]].append(sent)
        
        # Shuffle the captions
        for img_id in temp_dict:
            shuffle(temp_dict[img_id])

        # Write dataset to CSV file
        if dataset == COCO_TRAIN:
            fname = TRAIN_DATA
        elif dataset == COCO_VAL:
            fname = VAL_DATA
        with open(fname, 'w') as csv_file:
            writer = csv.writer(csv_file)
            count = 0
            for img_id, caption in temp_dict.items():
                count += 2
                if dataset == COCO_TRAIN:
                    # expand vocab
                    for sent in [caption[0], caption[1], caption[2], caption[3]]:
                        vocab.prep_train_seq(sent.rstrip())
                writer.writerow([caption[0].rstrip(),caption[1].rstrip()])
                writer.writerow([caption[2].rstrip(),caption[3].rstrip()])
            total += count

        # Update total/present the data split
        if dataset == COCO_TRAIN:
            split = count/total

    return split

def handle_quora(split, vocab):
    """ Read and process the raw Quora dataset
    Args:
        split: The split for train/val
    """

    print("Processing QUORA dataset...")
    with open(QUORA_RAW,'r') as quoraw, open(TRAIN_DATA, 'a+') as traincsv, open(VAL_DATA, 'a+') as valcsv:
        quoraw = csv.reader(quoraw, delimiter='\t')
        traincsv = csv.writer(traincsv)
        valcsv = csv.writer(valcsv)

        # Make the train/val split
        train_split = int(split*149263) # no of examples for training dataset(Quora contains 149263 pair matches)

        # Weed out the nonmatching pairs and add it to the appropriate csv file
        count = 0
        for idx, row in enumerate(quoraw):
            if idx == 0:
                continue
            is_pair = int(row[5])
            example = [row[3].rstrip(),row[4].rstrip()]
            if is_pair and (count <= train_split):
                count += 1
                # expand vocab
                for sent in example:
                    vocab.prep_train_seq(sent)
                traincsv.writerow(example)
            elif is_pair:
                count += 1
                valcsv.writerow(example)

def preprocess(vocab, max_keep=None):
    """ Prepare datasets, build vocab and write to tfRecords
    Args:
        vocab: vocab object to build vocab
        max_keep: The maximum number of words to keep. If None, it will write all words in the vocab dict to file.
    """

    # process coco dataset
    coco_split = handle_coco(vocab)

    # process quroa dataset
    handle_quora(coco_split, vocab)

    # save the vocabulary to the processed directory
    vocab.save_vocab(PROCESSED_DIR, max_keep=max_keep)

    print("Reading datasets and making tfRecords")
    handler = Dataset(vocab)
    for data, record in [(TRAIN_DATA, TRAIN_RECORD), (VAL_DATA, VAL_RECORD)]:
        handler.dataset_to_example(data, record)

    return


if __name__ == "__main__":
    # Download and unzip datasets
    for url in [QUORA,MSCOCO]:
        fp = download_file(url, RAW_DIR)
        if url == MSCOCO:
            print("UNZIPPING FROM: ", fp)
            zip_handler(fp, RAW_DIR)
    
    # Download and unzip fasttext
    fast_text = download_file(FASTTEXT, EXTERNAL_DIR)
    print("Unzipping Fasttext...")
    with zipfile.ZipFile(fast_text, "r") as zip_ref:
        zip_ref.extractall(EXTERNAL_DIR)
    os.remove(fast_text)

    # clean all the data and build vocab
    vocab_handler = Vocab(PROCESSED_DIR)
    preprocess(vocab_handler)

    # clean up extra files
    for file in [COCO_TRAIN,COCO_VAL,QUORA_RAW]:
        os.remove(file)
