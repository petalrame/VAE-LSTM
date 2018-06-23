""" Download data and pretrained vectors and split into Train/Val """
import csv
import glob
import json
import math
import os
import re
import shutil
import zipfile
from collections import defaultdict
from random import shuffle

import requests
from tqdm import tqdm

# put your data here
BASE_DIR = os.getcwd()
QUORA = "http://qim.ec.quoracdn.net/quora_duplicate_questions.tsv"
MSCOCO = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
FASTTEXT = "https://s3-us-west-1.amazonaws.com/fasttext-vectors/crawl-300d-2M.vec.zip"
COCO_TRAIN = BASE_DIR + "/captions_train2014.json"
COCO_VAL = BASE_DIR + "/captions_val2014.json"
QUORA_RAW = BASE_DIR + "/quora_duplicate_questions.tsv"

def download_file(url):
    """ General purpose function to download a file to working dir.
    Args:
        url: URL of the resource to download
    Returns:
        file_path: The path where the file was downloaded
    """
    
    # No existing file, let's download this file
    print("Downloading file located at:", url)
    
    # Make request for file and get content length for tqdm
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    wrote = 0

    # Use content disposition to fetch filename
    if 'Content-Disposition' in response.headers:
        d = response.headers.get('Content-Disposition')
        fname = re.findall("filename=(.+)", d)
        file_path = os.getcwd()+"/"+fname
    # Use splitting to find the filename
    elif url.find('/'):
        fname = url.rsplit('/',1)[1]
        file_path = os.getcwd()+"/"+fname
    print("Saving download to:", file_path)

    # Check to see if we have already downloaded to this location
    if os.path.exists(file_path):
        print("Actually, " + file_path + ", already exists. Skipping...")
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

def zip_handler(zipf):
    """ Handle the unzipping, moving and deleting of a specific dataset.
    NOTE: This function heavily depends on the dataset being handled and should be changed from project to project!
    Args:
        zipf: A zip file to unpack and process
    """

    # Unzip the file
    with zipfile.ZipFile(zipf, "r") as zip_ref:
        zip_ref.extractall(BASE_DIR)

    # For the MSCOCO dataset(not needed for Quora)
    del_dir = BASE_DIR+"/annotations"

    # Find files to keep
    keep_files = glob.glob(os.path.join(del_dir,'captions_*.json'))
    print("Keeping the following files:", keep_files)

    # Move the files to keepinto the working dir
    for file in keep_files:
        shutil.move(file, BASE_DIR)

    # Cleanup directory and zip file
    shutil.rmtree(del_dir)
    os.remove(zipf)

    return

def handle_coco():
    """ Handles all parsing of the raw MSCOCO dataset.
    This includes getting keeping only 4 captions per image and writing the dataset into CSV

    Returns:
        idx/total: Split from validation to train data
    """
    total = 0
    split = 0.0

    print("Processing MSCOCO dataset...")
    for dataset in [COCO_VAL, COCO_TRAIN]:
        temp_dict = defaultdict(list) # we store the data here for refinement
        # Read and parse the JSON
        with open(dataset) as f:
            data = json.load(f)
            for idx, anno in enumerate(data["annotations"]):
                sent = anno["caption"].rstrip()
                temp_dict[anno["image_id"]].append(sent)
        
        # Shuffle the captions
        for img_id, captions in temp_dict.items():
            shuffle(temp_dict[img_id])

        # Write dataset to CSV file
        if dataset == COCO_TRAIN:
            fname = "train_data.csv"
        elif dataset == COCO_VAL:
            fname = "val_data.csv"
        with open(fname, 'w') as csv_file:
            writer = csv.writer(csv_file)
            count = 0
            for img_id, caption in temp_dict.items():
                count += 2
                writer.writerow([caption[0],caption[1]])
                writer.writerow([caption[2],caption[3]])
            total += count

        # Update total/present the data split
        if dataset == COCO_TRAIN:
            split = count/total

    return split

def clean_data():
    """ Prepares the Quora and COCO datasets for readibility and processing
    """

    coco_split = handle_coco()

    print("Processing QUORA dataset...")
    with open(QUORA_RAW,'r') as quoraw, open('train_data.csv', 'a+') as traincsv, open('val_data.csv', 'a+') as valcsv:
        quoraw = csv.reader(quoraw, delimiter='\t')
        traincsv = csv.writer(traincsv)
        valcsv = csv.writer(valcsv)

        # Make the train/val split
        train_split = int(coco_split*149263) # no of examples for training dataset(Quora contains 149263 pair matches)

        # Weed out the nonmatching pairs and add it to the appropriate csv file
        count = 0
        for idx, row in enumerate(quoraw):
            if idx == 0:
                continue
            is_pair = int(row[5])
            if is_pair and (count <= train_split):
                count += 1
                traincsv.writerow([row[3].rstrip(),row[4].rstrip()])
            elif is_pair:
                count += 1
                valcsv.writerow([row[3].rstrip(),row[4].rstrip()])

    return


if __name__ == "__main__":
    for url in [QUORA,MSCOCO,FASTTEXT]:
        fp = download_file(url) #download
        if url == MSCOCO:
            print("UNZIPPING FROM: ", fp)
            zip_handler(fp)
        elif url == FASTTEXT:
            print("Unzipping Fasttext...")
            with zipfile.ZipFile(fp, "r") as zip_ref:
                zip_ref.extractall(BASE_DIR)
            os.remove(fp)

    # clean all the data
    clean_data()

    # clean up extra files
    for file in [COCO_TRAIN,COCO_VAL,QUORA_RAW]:
        os.remove(file)
