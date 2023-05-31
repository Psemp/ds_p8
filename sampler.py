import glob
import os
import math
import random
import shutil
import sys

from pathlib import Path


TEST_FOLDER = "data/fruits-360_dataset/fruits-360/Test/"
SAMPLE_FOLDER = "data/Test1/"
subfolders = glob.glob(pathname=f"{TEST_FOLDER}*", recursive=True)


def aggregate_folders(folder_list):
    """
    Just overcomplicating the selection process :
    getting keys as path/to/key whatever
    assigning path/to/key rest of path to a dict where key = path

    args :
    - folder_list : list of paths to aggregate

    returns
    - label_dict : dict with key as generic label and values as list of paths
    for label
    """
    unduplicated = []

    for folder in subfolders:
        fullpath = folder.split()[0]
        folder_name = os.path.basename(fullpath)
        if folder_name not in unduplicated:
            unduplicated.append(folder_name)

    label_dict = dict.fromkeys(unduplicated)

    for label_name in label_dict:
        for folder in folder_list:
            folder_splitpath = folder.split()[0]
            if os.path.basename(folder_splitpath) == label_name:
                if label_dict[label_name] is None:
                    label_dict[label_name] = [folder]
                else:
                    label_dict[label_name].append(folder)

    return label_dict


def select_random_files(directory_path, n):
    """
    Returns n randomly picked file in the directory defined as directory_path

    Args:
    - directory_path : path to the directory in which selection is performed
    - n : the number of files to select

    Returns :
    - random_files : list of randomly selected paths
    """

    files = list(Path(directory_path).glob('*'))
    random_filepaths = random.sample(files, min(n, len(files)))
    random_files = [str(path) for path in random_filepaths]
    return random_files


def check_paths_dict(paths_dict, n):
    """
    Returns keys of the dictionary which have less than n files in their directories

    Args:
    - paths_dict : dictionary of lists of directories
    - n : the threshold number of files

    Returns:
    - keys : list of keys which have less than n files in their directories
    """
    keys = []
    for key, directories in paths_dict.items():
        total_files = sum(len(list(Path(dir).glob('*'))) for dir in directories)
        if total_files < n:
            if key not in keys:
                keys.append(key)
    return keys


def copy_files_to_sample_folder(paths_dict, destination_folder):
    """
    Copies files from paths_dict to the sample folder while preserving the original structure.

    Args:
    - paths_dict : dictionary where keys are subdirectories of destination_folder and values are lists of file paths
    - destination_folder : path to the destination folder
    """

    for subdir, file_paths in paths_dict.items():
        dest_dir = os.path.join(destination_folder, subdir)
        os.makedirs(dest_dir, exist_ok=True)
        for file_path in file_paths:
            shutil.copy2(file_path, dest_dir)


label_dict = aggregate_folders(folder_list=subfolders)

sample_by_fruit = math.ceil((300 / label_dict.__len__()))

print(f"sample by category : {sample_by_fruit} images")

# rm -rf then recreate
if os.path.exists(SAMPLE_FOLDER):
    shutil.rmtree(SAMPLE_FOLDER)

os.makedirs(SAMPLE_FOLDER)

if check_paths_dict(paths_dict=label_dict, n=sample_by_fruit) == []:
    pass
else:
    print("sample too high, adapt sampling strategy")
    sys.exit()

image_samples = dict.fromkeys(label_dict.keys())

for category in image_samples:
    if sum(len(files) for _, _, files in os.walk(label_dict[category][0])) >= 5:
        image_samples[category] = select_random_files(directory_path=label_dict[category][0], n=sample_by_fruit)

copy_files_to_sample_folder(paths_dict=image_samples, destination_folder=SAMPLE_FOLDER)

print("Sampling : Done")
