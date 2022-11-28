import math
import numpy as np
import json
import sys
import os

PATH_TO_VIDEOPOSE = input("Enter the absolute path to your video_pose directory:")

existing_body_parts = {
    "nose":0, 
    "neck":1,
    "right_shoulder":2,
    "right_elbow":3,
    "right_wrist":4,
    "left_shoulder":5,
    "left_elbow":6,
    "left_wrist":7,
    "right_hip":8,
    "right_knee":9,
    "right_ankle":10,
    "left_hip":11,
    "left_knee":12,
    "left_ankle":13,
    "left_eye":14,
    "right_eye":15,
    "left_ear":16,
    "right_ear":17
}

coco_body_parts = {
    0:"nose",
    1:"left_eye",
    2:"right_eye",
    3:"left_ear",
    4:"right_ear",
    5:"left_shoulder",
    6:"right_shoulder",
    7:"left_elbow",
    8:"right_elbow",
    9:"left_wrist",
    10:"right_wrist",
    11:"left_hip",
    12:"right_hip",
    13:"left_knee",
    14:"right_knee",
    15:"left_ankle",
    16:"right_ankle"
}

OS_DIR = ["train", "validate", "test"]

annotation_id = 0  # increment this to have unique id for each annotation
for video_dir_name in OS_DIR:
    print(f"Creating JSONs with 17 keypoints in {video_dir_name}:")
    file_path = f"{PATH_TO_VIDEOPOSE}/{video_dir_name}/{video_dir_name}-coco.json"
    f = open(file_path)
    data = json.load(f)
    f.close()

    for player in data["annotations"] :
        playerKeypoints = player["keypoints"]

        nosex = playerKeypoints[0]
        nosey = playerKeypoints[1]
        noseCertainty = playerKeypoints[2]
        neckx = playerKeypoints[3]
        necky = playerKeypoints[4]

        # don't calculate half face distance if nose or neck is out of frame
        if (not ((nosex == 0 and nosey == 0) or (neckx == 0 and necky == 0))):
            deltaX = nosex - neckx
            deltaY = nosey - necky

            eye1X = nosex + 1/4*deltaY
            eye1Y = nosey + 1/4*deltaX
            eye2X = nosex - 1/4*deltaY
            eye2Y = nosey - 1/4*deltaX
            ear1X = nosex + 1/2*deltaY
            ear1Y = nosey + 1/2*deltaX
            ear2X = nosex - 1/2*deltaY
            ear2Y = nosey - 1/2*deltaX  

            eye1 = [eye1X, eye1Y, noseCertainty]
            eye2 = [eye2X, eye2Y, noseCertainty]
            ear1 = [ear1X, ear1Y, noseCertainty]
            ear2 = [ear2X, ear2Y, noseCertainty]

            #shoulderDeltaX = rightShoulderX - leftShoulderX. If it's >0, left shoulder has lower x value,
            #so left eye/ear will have lower x value
            #Add new keypoints to playerKeypoints in the order: left_eye, right_eye, left_ear, right_ear
            if(playerKeypoints[15] < playerKeypoints[6]):
                if (eye1X < eye2X):
                    playerKeypoints.extend(eye1)
                    playerKeypoints.extend(eye2)
                    playerKeypoints.extend(ear1)
                    playerKeypoints.extend(ear2)
                else:
                    playerKeypoints.extend(eye2)
                    playerKeypoints.extend(eye1)
                    playerKeypoints.extend(ear2)
                    playerKeypoints.extend(ear1)
            else:
                if (eye1X > eye2X):
                    playerKeypoints.extend(eye1)
                    playerKeypoints.extend(eye2)
                    playerKeypoints.extend(ear1)
                    playerKeypoints.extend(ear2)
                else:
                    playerKeypoints.extend(eye2)
                    playerKeypoints.extend(eye1)
                    playerKeypoints.extend(ear2)
                    playerKeypoints.extend(ear1)
        else:
            #If nose or neck is out of frame, just set all face keypoints to 0s
            playerKeypoints.extend([0,0,0,0,0,0,0,0,0,0,0,0])

        newPlayerKeypoints = []
        for i in range(17):
            cocoBodyPart = coco_body_parts.get(i)
            existingIndex = existing_body_parts.get(cocoBodyPart)
            newPlayerKeypoints.append(playerKeypoints[existingIndex * 3])
            newPlayerKeypoints.append(playerKeypoints[existingIndex * 3 + 1])
            newPlayerKeypoints.append(playerKeypoints[existingIndex * 3 + 2])

        player["keypoints"] = newPlayerKeypoints

    
    # Write the json to a file and save it
    outputFilenameString = f"{PATH_TO_VIDEOPOSE}/{video_dir_name}/{video_dir_name}-17-keypoints-coco.json"
    outputFile = open(outputFilenameString, "w")
    outputJson = json.dump(data, outputFile, indent=4)
    outputFile.close()