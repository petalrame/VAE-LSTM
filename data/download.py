""" Download and clean the raw dataset into Train/Test/Eval """
import math
import os
import re
import requests
from tqdm import tqdm

BASE_DIR = os.getcwd() # Where to store the train/test/eval files
QUORA = "http://qim.ec.quoracdn.net/quora_duplicate_questions.tsv"
COCO = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"

def download_file(url, file_path=None):
    """ General purpose function to download a file to a given file path
    Args:
        url: URL of the resource to download
        file_path: Where you want to save the download to
    """

    # Check for existing file
    if file_path and os.path.exists(file_path):
        print("File, " + file_path + ", already exists. Skipping...")
        return
    
    # No existing file, let's download this file
    print("Downloading file located at:", url)
    
    # Make request for file and get content length for tqdm
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    wrote = 0

    # Check to see if file_path is specified
    if file_path is None:
        # Use content disposition to fetch filename
        if 'Content-Disposition' in response.headers:
            d = response.headers.get('Content-Disposition')
            fname = re.findall("filename=(.+)", d)
            file_path = os.getcwd()+"/"+fname
        # Use splitting to find the filename
        elif url.find('/'):
            fname = url.rsplit('/',1)[1]
            file_path = os.getcwd()+"/"+fname
        print("Unspecified file path. Saving download to:", file_path)

    # Show progress of download and write to file
    with open(file_path, 'wb') as f:
        for chunk in tqdm(response.iter_content(chunk_size=1024), total=total_size/1024, unit='KB'):
            if chunk:
                wrote += len(chunk)
                f.write(chunk)

    # Some error handling
    if total_size != 0 and wrote != total_size:
        raise ValueError('Did not download the full file. Something went wrong')

    return

