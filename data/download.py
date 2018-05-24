""" Download and clean the raw dataset into Train/Test/Eval """
import glob
import math
import os
import re
import shutil
import zipfile

import requests
from tqdm import tqdm

BASE_DIR = os.getcwd()
QUORA = "http://qim.ec.quoracdn.net/quora_duplicate_questions.tsv"
MSCOCO = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"

def download_file(url, file_path=None):
    """ General purpose function to download a file to a given file path.
    Args:
        url: URL of the resource to download
        file_path: Where you want to save the download to(Default: Save to working dir with name provided in URL)
    Returns:
        file_path: The path where the file was downloaded
    """
    
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

    # For the MSCOCO dataset
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

if __name__ == "__main__":
    fp = download_file(MSCOCO)
    zip_handler(fp)  
