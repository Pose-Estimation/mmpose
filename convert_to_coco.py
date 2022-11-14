import os
import json
import re
from PIL import Image
import numpy as np

PATH_TO_VIDEOPOSE = input("Enter the absolute path to your video_pose directory:")
VIDEO_POSE_TYPES = {"No_penalty": 0, "Slashing": 1, "Tripping": 2}

annotation_id = 0  # increment this to have unique id for each annotation
for video_dir_name in os.listdir(PATH_TO_VIDEOPOSE):
    video_dir_full_path = os.path.join(PATH_TO_VIDEOPOSE, video_dir_name)

    if (
        not os.path.isfile(video_dir_full_path)
        and video_dir_name in VIDEO_POSE_TYPES.keys()
    ):
        for game_dir_name in os.listdir(video_dir_full_path):
            game_dir_full_path = os.path.join(video_dir_full_path, game_dir_name)
            if not os.path.isfile(game_dir_full_path):
                opened_file = open(f"{game_dir_full_path}/{game_dir_name}.json")
                json_file = json.load(opened_file)

                coco_dict = {}
                coco_dict["images"] = []
                coco_dict["annotations"] = []
                coco_dict["categories"] = []

                # images
                for game_file in os.listdir(game_dir_full_path):
                    if game_file.endswith(".png"):
                        id = int(re.search("[0-9]*(?=\.png)", game_file).group(0))

                        # need to pip install pillow
                        img = Image.open(f"{game_dir_full_path}/{game_file}")
                        width = img.width
                        height = img.height

                        image_dict = {
                            "file_name": game_file,
                            "height": height,
                            "width": width,
                            "id": id,
                        }
                        coco_dict["images"].append(image_dict)

                # annotations and categories
                image_id = 0
                checked_categories = False
                for frame in json_file:
                    for key, value in frame.items():

                        if key == "frameNum":
                            image_id = value
                        else:
                            if not checked_categories:
                                coco_dict["categories"].append(
                                    {
                                        "id": int(key[1:]),
                                        "name": "person",
                                    }
                                )

                            annotation = {}

                            annotation["keypoints"] = []
                            keypoints = np.split(np.array(value), len(value) // 3)
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
                            annotation["category_id"] = int(key[1:])
                            annotation["id"] = annotation_id

                            annotation_id += 1

                            coco_dict["annotations"].append(annotation)

                    checked_categories = True

                with open(
                    f"{game_dir_full_path}/{game_dir_name}-coco.json", "w"
                ) as outfile:
                    json.dump(coco_dict, outfile, indent=4)
