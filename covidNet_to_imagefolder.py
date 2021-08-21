import os
from posixpath import splitdrive
import re

def create_imagefolder_dataset(txt_file, target_name = "COVIDNet_ImageFolder", image_source = "covid_data"):
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
        for line in f:
            splitted = line.split()
            # verify we have the right amount of information
            if len(splitted) != 4:
                print(splitted)
                continue

            _, image_, class_, _ = splitted

            # dirs for classes
            class_dir = os.path.join(split_dir, class_)
            if not os.path.isdir(class_dir):
                os.makedirs(class_dir)
                print("Created class-directory")

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

if __name__ == "__main__":
    create_imagefolder_dataset("train_split.txt")
    create_imagefolder_dataset("test_split.txt")
