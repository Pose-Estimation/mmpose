import os
import json
import re
from PIL import Image
import numpy as np
import pandas as pd

PATH_TO_VIDEOPOSE = input(
    "Enter the absolute path to your video_pose directory:")
VIDEO_POSE_TYPES = {
    "No_penalty": {
        "slow": [
            450, 474, 167, 175, 494, 353, 354, 532, 558, 570, 625, 232, 56,
            135, 141, 146, 429, 588
        ],
        "train":
        23,
        "validate":
        5,
        "test":
        5,
        "total":
        33
    },
    "Slashing": {
        "slow": [
            305, 217, 245, 45, 306, 307, 336, 515, 218, 241, 391, 409, 306,
            323, 7, 20, 82, 87, 127, 143, 149, 139, 413
        ],
        "train":
        19,
        "validate":
        4,
        "test":
        4,
        "total":
        27
    },
    "Tripping": {
        "slow": [149, 13, 471, 503, 522, 215, 235, 382, 408, 70, 6, 137],
        "train": 14,
        "validate": 3,
        "test": 3,
        "total": 20
    }
}
OS_DIR = ["train", "validate", "test"]


def create_coco_dict():
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


def get_json_type(frame_id, train_list, validate_list):
    if frame_id in train_list:
        return "train"
    elif frame_id in validate_list:
        return "validate"

    return "test"


def main():
    annotation_id = 0  # increment this to have unique id for each annotation

    splits = {
        "train": create_coco_dict(),
        "validate": create_coco_dict(),
        "test": create_coco_dict()
    }

    np.random.seed(0)

    image_id = 0

    filtered_videos = pd.read_csv("C:/Users/stavro/Desktop/capstone/filtered_videos.csv")

    for data_dir in OS_DIR:
        if not os.path.isdir(f'{PATH_TO_VIDEOPOSE}/{data_dir}'):
            os.mkdir(f'{PATH_TO_VIDEOPOSE}/{data_dir}')

    for video_dir_name in os.listdir(PATH_TO_VIDEOPOSE):
        video_dir_full_path = os.path.join(PATH_TO_VIDEOPOSE, video_dir_name)
        if (not os.path.isfile(video_dir_full_path)
                and video_dir_name in VIDEO_POSE_TYPES):
            print(
                f"\nConverting custom jsons in {video_dir_name} directory to coco format:"
            )

            slowmo = VIDEO_POSE_TYPES[video_dir_name]["slow"]
            games = filtered_videos[video_dir_name].to_list()

            start_id = image_id
            indices = np.arange(VIDEO_POSE_TYPES[video_dir_name]["total"])
            np.random.shuffle(indices)

            train_index = VIDEO_POSE_TYPES[video_dir_name]["train"]
            train_list = indices[:train_index]
            val_list = indices[train_index:train_index +
                               VIDEO_POSE_TYPES[video_dir_name]["validate"]]
            frame_index = 0
            for game_dir_name in os.listdir(video_dir_full_path):
                game_dir_full_path = os.path.join(video_dir_full_path,
                                                  game_dir_name)

                if not os.path.isfile(
                        game_dir_full_path) and game_dir_name in games:
                    game_number = re.search("[0-9]{2,3}$", game_dir_name)
                    is_slowed = int(game_number.group(
                        0)) in slowmo if game_number else False

                    json_type = get_json_type(frame_index, train_list, val_list)
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
                                continue

                            # need to pip install pillow
                            img = Image.open(
                                f"{game_dir_full_path}/{game_file}")
                            width = img.width
                            height = img.height
                            img.save(
                                f"{PATH_TO_VIDEOPOSE}/{json_type}/{temp_id}.png"
                            )

                            image_dict = {
                                "file_name": f'{temp_id}.png',
                                "height": height,
                                "width": width,
                                "id": temp_id,
                            }
                            splits[json_type]["images"].append(image_dict)
                            temp_id += 1

                    # annotations
                    for frame in json_file:

                        if is_slowed and not ((image_id - start_id) % 4 == 0):
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
                                # Only need 1 category for person?
                                annotation["category_id"] = 1
                                annotation["id"] = annotation_id

                                annotation_id += 1

                                splits[json_type]["annotations"].append(
                                    annotation)
                        image_id += 1
                    
                    frame_index +=1
                        # checked_categories = True
            print(
                f"\nFinished Processing images for {video_dir_name} at index {image_id}"
            )

    for key, value in splits.items():
        coco_file_name = f"{PATH_TO_VIDEOPOSE}/{key}/{key}-coco.json"
        with open(coco_file_name, "w") as outfile:
            json.dump(value, outfile, indent=4)
            outfile.close()
        print(f"-> Generated {coco_file_name}")

    print("DONE")


if __name__ == "__main__":
    main()