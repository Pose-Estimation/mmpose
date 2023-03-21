import math
import os
import json
import re
from typing import List
from PIL import Image
import numpy as np
import pandas as pd

PATH_TO_VIDEOPOSE = input(
    "Enter the absolute path to your video_pose directory:")
VIDEO_POSE_TYPES = {
    "No_penalty": {
        "slow": [
            450,
            474,
            167,
            175,
            494,
            353,
            354,
            532,
            558,
            570,
            625,
            232,
            56,
            135,
            141,
            146,
            429,
            588,
        ],
    },
    "Slashing": {
        "slow": [
            305,
            217,
            245,
            45,
            306,
            307,
            336,
            515,
            218,
            241,
            391,
            409,
            306,
            323,
            7,
            20,
            82,
            87,
            127,
            143,
            149,
            139,
            413,
        ],
    },
    "Tripping": {
        "slow": [149, 13, 471, 503, 522, 215, 235, 382, 408, 70, 6, 137],
    },
}

# List of directories for full dataset
OS_DIR = ["full_data", "train", "validate", "test"]

TRAINING_PERCENTAGE = 0.70
VALIDATION_PERCENTAGE = 0.10

image_mapping = {}


def create_coco_dict() -> dict:
    """
    Method that creates a dictionary object with the COCO format
    https://cocodataset.org/#format-data
    """
    coco_dict = {}
    coco_dict["images"] = []
    coco_dict["annotations"] = []
    coco_dict["categories"] = [{
        "supercategory":
        "person",
        "id":
        1,
        "name":
        "person",
        "keypoints": [
            "head",
            "neck",
            "right_shoulder",
            "right_elbow",
            "right_wrist",
            "left_shoulder",
            "left_elbow",
            "left_wrist",
            "right_hip",
            "right_knee",
            "right_ankle",
            "left_hip",
            "left_knee",
            "left_ankle",
            # "hockey_grip",
            # "hockey_hill",
        ],
        "skeleton": [
            [13, 12],  # left ankle - left knee
            [12, 11],  # left knee - left hip
            [10, 9],  # right ankle - right knee
            [9, 8],  # right knee - right hip
            [11, 8],  # left hip - right hip
            [5, 11],  # left shoulder - left hip
            [2, 8],  # right shoulder - right hip
            [5, 2],  # left shoulder - right shoulder
            [5, 6],  # left shoulder - left elbow
            [2, 3],  # right shoulder - right elbow
            [6, 7],  # left elbow - left wrist
            [3, 4],  # right elbow - right wrist
            [1, 5],  # neck - left shoulder
            [1, 2],  # neck - right shoulder
            [1, 0],  # neck - head
            # [14, 15],  # hockey grip - hockey hill
        ],
    }]
    return coco_dict


def get_json_type(frame_id: int, train_list: List[int],
                  validate_list: List[int]):
    """
    Method that matches an index to a set type (train/validate/test)
    """
    if frame_id in train_list:
        return "train"
    elif frame_id in validate_list:
        return "validate"

    return "test"


def main():
    annotation_id = 0  # increment this to have unique id for each annotation

    # Create dictionary for each set type
    splits = {
        "train": create_coco_dict(),
        "validate": create_coco_dict(),
        "test": create_coco_dict(),
    }

    image_id = 0

    # Only use videos with no fans included
    filtered_videos = pd.read_csv("C:/Users/stavro/Desktop/capstone/filtered_videos.csv")

    # Create new directories if they do not already exist
    for data_dir in OS_DIR:
        if not os.path.isdir(
                f"{PATH_TO_VIDEOPOSE}/{data_dir}") and not os.path.isdir(
                    f"{PATH_TO_VIDEOPOSE}/full_data/{data_dir}"):
            if data_dir == "full_data":
                os.mkdir(f"{PATH_TO_VIDEOPOSE}/{data_dir}")
            else:
                os.mkdir(f"{PATH_TO_VIDEOPOSE}/full_data/{data_dir}")

    # Iterate through each penalty type
    for video_dir_name in os.listdir(PATH_TO_VIDEOPOSE):
        video_dir_full_path = os.path.join(PATH_TO_VIDEOPOSE, video_dir_name)
        if (not os.path.isfile(video_dir_full_path)
                and video_dir_name in VIDEO_POSE_TYPES):
            print(
                f"\nConverting custom jsons in {video_dir_name} directory to coco format:"
            )

            # Retrieve slow motion videos and filter out games
            slowmo = VIDEO_POSE_TYPES[video_dir_name]["slow"]
            games = filtered_videos[video_dir_name].dropna().to_list()

            # Add games with augmentations
            augmented_games = games.copy()
            n = 0
            for i, game in enumerate(games):
                augmented_games.insert(i * n + 1, f'{game}100')
                augmented_games.insert(i * n + 2, f'{game}200')
                n += 3
            games = augmented_games

            # Randomly assign indices to different sets
            game_count = len(games)
            indices = np.arange(game_count)
            np.random.seed(0)
            np.random.shuffle(indices)
            train_index = math.floor(TRAINING_PERCENTAGE * game_count)
            train_list = indices[:train_index]
            val_list = indices[train_index:math.floor((VALIDATION_PERCENTAGE +
                                                       TRAINING_PERCENTAGE) *
                                                      game_count)]

            start_id = image_id
            frame_index = 0

            # Iterate through each game folder
            for game_dir_name in os.listdir(video_dir_full_path):
                game_dir_full_path = os.path.join(video_dir_full_path,
                                                  game_dir_name)

                if not os.path.isfile(
                        game_dir_full_path) and game_dir_name in games:
                    # Verify if the current video is slowed down
                    game_number = re.search("[0-9]{2,3}$", game_dir_name)
                    is_slowed = (
                        int(game_number.group(0)) in slowmo
                        if game_number else False)

                    json_type = get_json_type(frame_index, train_list,
                                              val_list)
                    print(f"Converting {game_dir_full_path}...")
                    
                    opened_file = open(
                        f"{game_dir_full_path}/{game_dir_name}.json")
                    json_file = json.load(opened_file)

                    # images
                    temp_id = image_id
                    for game_file in os.listdir(game_dir_full_path):
                        if game_file.endswith(".png"):
                            if is_slowed and not (
                                (temp_id - start_id) % 4 == 0):
                                temp_id += 1
                                continue

                            # need to pip install pillow
                            img = Image.open(
                                f"{game_dir_full_path}/{game_file}")
                            width = img.width
                            height = img.height
                            # Copy image to new directory
                            img.save(
                                f"{PATH_TO_VIDEOPOSE}/full_data/{json_type}/{temp_id}.png"
                            )

                            # Save image mapping
                            if game_dir_name in image_mapping:
                                image_mapping[game_dir_name].append(temp_id)
                            else:
                                image_mapping[game_dir_name] = [temp_id]

                            image_dict = {
                                "file_name": f"{temp_id}.png",
                                "height": height,
                                "width": width,
                                "id": temp_id,
                            }
                            splits[json_type]["images"].append(image_dict)
                            temp_id += 1

                    # annotations
                    for frame in json_file:
                        # Take every fourth image if video is slowed down
                        if is_slowed and not ((image_id - start_id) % 4 == 0):
                            image_id += 1
                            continue

                        for key, value in frame.items():
                            if not (key == "frameNum"):
                                annotation = {}

                                annotation["keypoints"] = []
                                keypoints = np.split(
                                    np.array(value),
                                    len(value) // 3)[:14]
                                num_keypoints = 0
                                for x, y, _ in keypoints:
                                    if x == y == 0:
                                        v = 0
                                    else:
                                        v = 2
                                        num_keypoints += 1

                                    annotation["keypoints"].extend([x, y, v])

                                annotation["num_keypoints"] = num_keypoints
                                annotation["image_id"] = image_id
                                annotation["category_id"] = 1
                                annotation["id"] = annotation_id

                                annotation_id += 1

                                splits[json_type]["annotations"].append(
                                    annotation)
                        image_id += 1

                    frame_index += 1

            print(
                f"\nFinished Processing images for {video_dir_name} at index {image_id}"
            )

    # Create json files
    for key, value in splits.items():
        coco_file_name = f"{PATH_TO_VIDEOPOSE}/full_data/{key}/{key}-coco.json"
        with open(coco_file_name, "w") as outfile:
            json.dump(value, outfile, indent=4)
            outfile.close()
        print(f"-> Generated {coco_file_name}")

    #Create mapping file
    with open(f'{PATH_TO_VIDEOPOSE}/full_data/image_mapping.json',
              "w") as outfile:
        json.dump(image_mapping, outfile, indent=4)
        outfile.close()

    print("DONE")


if __name__ == "__main__":
    main()