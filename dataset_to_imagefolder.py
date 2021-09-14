# script to convert chexpert to imagefolder

import os 
import pandas as pd
from tqdm import tqdm

def covidnet_create_imagefolder_dataset(txt_file, target_name = "COVIDNet_ImageFolder", image_source = "covid_data", binary = False):
    """
    Function to create a folderstructure possible to use for vissl
    """
    if not txt_file:
        raise("Please set a txt file.")

    split = txt_file.split("_")[0]

    dataset_dir = os.path.join(os.getcwd(), target_name)
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
        print("Created root-directory")

    split_dir = os.path.join(dataset_dir, split)
    if not os.path.isdir(split_dir):
        os.makedirs(split_dir)

    counter = 0 
    with open(txt_file, "r") as f:
        for line in tqdm(f, position=0):
            splitted = line.split()
            # verify we have the right amount of information
            if len(splitted) != 4:
                print(splitted)
                continue

            _, image_, class_, _ = splitted

            if binary:
                if not class_ == "COVID-19":
                    class_ = "NO_COVID-19"

            # dirs for classes
            class_dir = os.path.join(split_dir, class_)
            if not os.path.isdir(class_dir):
                os.makedirs(class_dir)
                tqdm.write("Created class-directory")

            # get full path to images
            src_image_path = os.path.join(os.getcwd(), image_source, split, image_)
            dst_image_path = os.path.join(class_dir, image_)

            # create links
            if not os.path.islink(dst_image_path):
                counter += 1
                os.symlink(src_image_path, dst_image_path)
            else:
                continue

    print(f"Created {counter} symlinks for {txt_file}") 


def chexpert_create_imagefolder_dataset(csv_file, image_source, target_name = "chexpert_ImageFolder"):
    df = pd.read_csv(csv_file, usecols=["Path", 
                                        "No Finding", 
                                        "Lung Opacity",
                                        "Lung Lesion",
                                        "Edema",
                                        "Pneumonia",
                                        "Atelectasis",
                                        "Pneumothorax",
                                        "Pleural Effusion",
                                        "Pleural Other"])

    if "train" in csv_file:
        split = "train"
    elif "valid" in csv_file:
        split = "test"

    dataset_dir = os.path.join(os.getcwd(), target_name)
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
        print("Created root-directory")

    split_dir = os.path.join(dataset_dir, split)
    if not os.path.isdir(split_dir):
        os.makedirs(split_dir)

    # iterate over all df rows
    counter = 0
    for i, line in tqdm(df.iterrows(), position=0):
        src_image_path = os.path.join(image_source, line[0])
        # replace / with _ for individual image names
        image_name = "_".join(line[0].split("/")[-3:])

        # iterate over all entries within the row to find the class
        for axis_label, entry in line.iteritems():
            if entry == 1.0:
                class_dir = os.path.join(split_dir, axis_label)
                if not os.path.isdir(class_dir):
                    os.makedirs(class_dir)
                    tqdm.write("Created class-directory")

                dst_image_path = os.path.join(class_dir, image_name)

                if not os.path.islink(dst_image_path):
                    counter += 1
                    os.symlink(src_image_path, dst_image_path)
                    continue
                else:
                    continue

    print(f"Created {counter} symlinks for {csv_file}")


if __name__ == "__main__":

    # covidnet
    #covidnet_create_imagefolder_dataset("train_split.txt")
    #covidnet_create_imagefolder_dataset("test_split.txt")

    # chexpert
    train_csv = "/export/scratch/sgimmini/chexpert/CheXpert-v1.0-small/train.csv"
    val_csv = "/export/scratch/sgimmini/chexpert/CheXpert-v1.0-small/valid.csv"

    chexpert_create_imagefolder_dataset(train_csv, image_source="/export/scratch/sgimmini/chexpert/")
    chexpert_create_imagefolder_dataset(val_csv, image_source="/export/scratch/sgimmini/chexpert/")