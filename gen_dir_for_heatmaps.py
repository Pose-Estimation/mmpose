import numpy as np
import json
import sys
import os
import shutil 


PATH_TO_VIDEOPOSE = input("Enter the absolute path to your video_pose directory:")
# copy the training set into the dir and name it train_heatmaps
video_dir_name = "train_heatmaps"
# make sure to rename json file to train_heatmaps-coco.json
train_json = "train_heatmaps"

# open json file for training set annotation
json_file_path = f"{PATH_TO_VIDEOPOSE}/{video_dir_name}/{train_json}-bbox-appended.json"
json_f = open(json_file_path)
data_json = json.load(json_f)
json_f.close()

# get all the images in the dataset
filelist= [file for file in os.listdir(f"{PATH_TO_VIDEOPOSE}/{video_dir_name}") if file.endswith('.png')]

print("Total number of images found: " + str(len(filelist)))

print("Moving them them to their own directories...")

for i in filelist:
  print("Name of file : " + str(i))
  print("Transferring...")
  os.mkdir(os.path.join(f"{PATH_TO_VIDEOPOSE}/{video_dir_name}" , i.split(".")[0]))
  shutil.move(os.path.join(f"{PATH_TO_VIDEOPOSE}/{video_dir_name}" , i), os.path.join(f"{PATH_TO_VIDEOPOSE}/{video_dir_name}" , i.split(".")[0]))

if len(filelist) == 0:
    print("Images already moved into directories.")

# get all the newly created directories 
img_directories = [ f.path for f in os.scandir(f"{PATH_TO_VIDEOPOSE}/{video_dir_name}") if f.is_dir() ]
print(str(len(img_directories)) + " directories found")

# parse through each directory
annotation_counter = 0
print("Creating annotation file for each directory")
for directory in img_directories:
    print("current directory: " + str(directory))

    coco_dict = {}
   
    # get current_image
    coco_dict["images"] = data_json["images"][annotation_counter]
    coco_dict["annotations"] = []

    # get annotations for the current image
    annotations = data_json["annotations"]
    for player in annotations:
        img_id = player["image_id"]

        if img_id == annotation_counter:
            coco_dict["annotations"].append(player)

    # get categories
    coco_dict["categories"] = data_json["categories"]

    annot_str = str(annotation_counter)
    outputFilenameString = f"{directory}/{annot_str}.json"
    print("Saving " + outputFilenameString)
    outputFile = open(outputFilenameString, "w")
    outputJson = json.dump(coco_dict, outputFile, indent=4)

    annotation_counter +=1
    













