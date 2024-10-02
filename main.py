from imutils import paths
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import splitfolders
from torch import nn
import numpy as np
import os
import wget
from PIL import Image
import cv2
import requests
import tarfile
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img

# TODO --------------STEP 1--------------
# def download_and_extract(url, destination_folder):
#     # Create the destination folder if it doesn't exist
#     if not os.path.exists(destination_folder):
#         os.makedirs(destination_folder)
#
#     # Get the filename from the URL
#     filename = os.path.join(destination_folder, url.split("/")[-1])
#
#     # Download the file
#     response = requests.get(url, stream=True)
#     with open(filename, 'wb') as file:
#         for chunk in response.iter_content(chunk_size=1024):
#             if chunk:
#                 file.write(chunk)
#
#     # Extract the contents of the tar.gz file
#     with tarfile.open(filename, 'r:gz') as tar:
#         tar.extractall(destination_folder)
#
#     # Remove the downloaded compressed file
#     os.remove(filename)
#
# dataset_url = "http://image.ntua.gr/iva/datasets/flickr_logos/flickr_logos_27_dataset.tar.gz"
#
destination_folder = "flickr_logos_dataset"
# download_and_extract(dataset_url, destination_folder)

# TODO --------------STEP 2--------------
# fname = "./flickr_logos_dataset/flickr_logos_27_dataset/flickr_logos_27_dataset_images.tar.gz"
#
# with tarfile.open(fname, 'r:gz') as tar:
#     tar.extractall(destination_folder)
#
# os.remove(fname)

# TODO --------------STEP 3--------------
txt_path = "./flickr_logos_dataset/flickr_logos_27_dataset/flickr_logos_27_dataset_training_set_annotation.txt"
df = pd.read_csv(txt_path,
                 sep='\s+',
                 header=None)
df.shape
columns = ['filename', 'class', 'sub-class','xmin', 'ymin', 'xmax', 'ymax']
df.columns = columns
df.head(5)
#
classes = df['class'].unique().tolist()
#
# print("Classess = ",classes)
# print("\n\nTotal Classess = ",len(classes))
#
class_mapping = {class_name: i for i, class_name in enumerate(classes)}
#
# for class_name, class_no in class_mapping.items():
#     print(f"{class_name}: {class_no}")

# TODO --------------STEP 4--------------
# def is_valid_image(img_path):
#     try:
#         Image.open(img_path).verify()
#         return True
#     except (IOError, SyntaxError):
#         return False
#
# def remove_broken_and_invalid_entries(folder_path, annotation_file_path):
#
#     total_images_before = len(os.listdir(folder_path))
#
#     with open(annotation_file_path, 'r') as file:
#         total_entries_before = len(file.readlines())
#
#     # Read the annotation file into a list
#     with open(annotation_file_path, 'r') as file:
#         annotations = file.readlines()
#
#     # Filter out broken and invalid entries
#     valid_annotations = []
#     for annotation in annotations:
#         parts = annotation.split()
#         img_name, class_name, _, xmin, ymin, xmax, ymax = parts
#
#         # Check if image is valid
#         img_path = os.path.join(folder_path, img_name)
#         if not is_valid_image(img_path):
#             continue
#
#         # Check if bounding box is valid
#         xmin, ymin, xmax, ymax = map(int, [xmin, ymin, xmax, ymax])
#         if xmin >= xmax or ymin >= ymax:
#             continue
#
#         # If both checks pass, add the annotation to the valid list
#         valid_annotations.append(annotation)
#
#     # Update the annotation file
#     with open(annotation_file_path, 'w') as file:
#         file.writelines(valid_annotations)
#
#     # Remove broken and invalid images
#     for root, dirs, files in os.walk(folder_path):
#         for file in files:
#             img_path = os.path.join(root, file)
#             if not is_valid_image(img_path):
#                 os.remove(img_path)
#             else:
#                 img_name = file
#                 annotation_exists = any(img_name in annotation for annotation in valid_annotations)
#                 if not annotation_exists:
#                     print(img_path)
#                     os.remove(img_path)
#
#     # Count the total number of images after removal
#     total_images_after = len(os.listdir(folder_path))
#
#     total_entries_after = len(valid_annotations)
#
#     print(f"Total number of entries before: {total_entries_before}")
#     print(f"Total number of entries after Removal: {total_entries_after}")
#
#     print(f"Total number of images before Removal: {total_images_before}")
#     print(f"Total number of images after Removal: {total_images_after}")
#
# folder_path = "./flickr_logos_dataset/flickr_logos_27_dataset_images"
# annotation_file_path = "./flickr_logos_dataset/flickr_logos_27_dataset/flickr_logos_27_dataset_training_set_annotation.txt"
# remove_broken_and_invalid_entries(folder_path, annotation_file_path)

# TODO --------------STEP 5--------------
df = pd.read_csv("./flickr_logos_dataset/flickr_logos_27_dataset/flickr_logos_27_dataset_training_set_annotation.txt",
                 sep='\s+',
                 header=None)
df.shape
df.columns = columns
df.head(5)

# TODO --------------STEP 6--------------
IMAGES_FOLDER_PATH = './flickr_logos_dataset/flickr_logos_27_dataset_images'
OUTPUT_FOLDER_PATH = 'LOGOS'

# Create folders
output_images_folder = os.path.join(OUTPUT_FOLDER_PATH, 'images')
output_labels_folder = os.path.join(OUTPUT_FOLDER_PATH, 'labels')
os.makedirs(output_images_folder, exist_ok=True)
os.makedirs(output_labels_folder, exist_ok=True)

# for idx, row in df.iterrows():
#     filename = row['filename']
#     class_name = row['class']
#     xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
#
#     # Read image
#     image_path = os.path.join(IMAGES_FOLDER_PATH, filename)
#     image = Image.open(image_path).convert("RGB")
#     image_w, image_h = image.size
#
#     # Calculate normalized bounding box coordinates
#     b_center_x = (xmin + xmax) / 2
#     b_center_y = (ymin + ymax) / 2
#     b_width = (xmax - xmin)
#     b_height = (ymax - ymin)
#
#     b_center_x /= image_w
#     b_center_y /= image_h
#     b_width /= image_w
#     b_height /= image_h
#
#     # Save image
#     output_image_path = os.path.join(output_images_folder, filename)
#     image.save(output_image_path)
#
#     # Save label file
#     label_filename = os.path.splitext(filename)[0] + '.txt'
#     label_path = os.path.join(output_labels_folder, label_filename)
#     with open(label_path, 'w') as label_file:
#         class_id = class_mapping[class_name]
#         label_file.write(f"{class_id} {b_center_x} {b_center_y} {b_width} {b_height}")
#
# print("Processing complete.")

label_path = './LOGOS/labels/2675240646.txt'
img = './LOGOS/images/2675240646.jpg'
image = cv2.imread(img)

# with open(label_path, "r") as f:
#     labels = f.read().strip().split("\n")
#     for label in labels:
#         class_id, x_center, y_center, width, height = map(float, label.split())
#         x_min = int((x_center - width / 2) * image.shape[1])
#         y_min = int((y_center - height / 2) * image.shape[0])
#         x_max = int((x_center + width / 2) * image.shape[1])
#         y_max = int((y_center + height / 2) * image.shape[0])
#         cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
#         plt.imshow(image)
#
# splitfolders.ratio('./LOGOS', output="data", seed=42, ratio=(0.8,0.2))

# TODO --------------STEP 7--------------
# from ultralytics import YOLO
# import yaml
# model = YOLO("yolov8n.pt")
# dict_classes = model.model.names
# # print(dict_classes)
# data = {'train' :  './data/train',
#         'val' :  './data/val',
#         'test' :  './data/val',
#         'nc': len(classes),
#         'names': classes
#         }

# file_path = 'data.yaml'
# with open(file_path, 'w') as f:
#     yaml.dump(data, f)
#
# # read the content in .yaml file
# with open('./data.yaml', 'r') as f:
#     hamster_yaml = yaml.safe_load(f)
#     # display(hamster_yaml)
#     print(hamster_yaml)

# data_path = './data.yaml'

# model.train(data=data_path, epochs=50, batch=32);
if __name__ == '__main__':
    from ultralytics import YOLO
    import yaml

    # Load the model and prepare the data
    model = YOLO("yolov8n.pt")
    dict_classes = model.model.names
    data = {
        'train': './data/train',
        'val': './data/val',
        'test': './data/val',
        'nc': len(classes),
        'names': classes
    }

    file_path = 'data.yaml'
    with open(file_path, 'w') as f:
        yaml.dump(data, f)

    # Train the model
    data_path = './data.yaml'
    model.train(data=data_path, epochs=50, batch=32)

