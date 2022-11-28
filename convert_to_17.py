import math
import numpy as np
import json
import sys
import os

PATH_TO_VIDEOPOSE = input("Enter the absolute path to your video_pose directory:")
VIDEO_POSE_TYPES = {"No_penalty": 0, "Slashing": 1, "Tripping": 2}

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
    "hockey_grip":14,
    "hockey_hill":15,
    "left_eye":16,
    "right_eye":17,
    "left_ear":18,
    "right_ear":19
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

annotation_id = 0  # increment this to have unique id for each annotation
for video_dir_name in os.listdir(PATH_TO_VIDEOPOSE):
    video_dir_full_path = os.path.join(PATH_TO_VIDEOPOSE, video_dir_name)

    if (
        not os.path.isfile(video_dir_full_path)
        and video_dir_name in VIDEO_POSE_TYPES.keys()
    ):

        print(f"Creating JSONs with 17 keypoints in {video_dir_name}:")
        for game_dir_name in os.listdir(video_dir_full_path):
            game_dir_full_path = os.path.join(video_dir_full_path, game_dir_name)
            if not os.path.isfile(game_dir_full_path):
                print(f"Converting {game_dir_full_path}...")

                f = open(f"{game_dir_full_path}/{game_dir_name}-bbox-appended.json")
                data = json.load(f)
                f.close()

                #get image width and height, assuming they are the same size for all images in one video
                imageWidth = data["images"][0]["width"]
                imageHeight = data["images"][0]["height"]

                for player in data["annotations"] :
                    playerNumberString = "p" + str(player["category_id"])
                    playerKeypoints = player["keypoints"]

                    nosex = playerKeypoints[0]
                    nosey = playerKeypoints[1]
                    neckx = playerKeypoints[3]
                    necky = playerKeypoints[4]

                    halfFaceDistance = 0
                    # don't calculate half face distance if nose or neck is out of frame
                    if (not ((nosex == 0 and nosey == 0) or (neckx == 0 and necky == 0))):
                        deltaX = nosex - neckx
                        deltaY = nosey - necky

                        eye1X = nosex + 1/4*deltaY
                        if (nosex == 482.758):
                            print(f'deltaX is {deltaX}, deltaY is {deltaY}, noseX is {nosex} so eye1X is {eye1X}')
                        eye1Y = nosey + 1/4*deltaX
                        eye2X = nosex - 1/4*deltaY
                        if (nosex == 482.758):
                            print(f'deltaX is {deltaX}, deltaY is {deltaY}, noseX is {nosex} so eye2X is {eye2X}')
                        eye2Y = nosey - 1/4*deltaX
                        ear1X = nosex + 1/2*deltaY
                        if (nosex == 482.758):
                            print(f'deltaX is {deltaX}, deltaY is {deltaY}, noseX is {nosex} so ear1X is {ear1X}')
                        ear1Y = nosey + 1/2*deltaX
                        ear2X = nosex - 1/2*deltaY
                        if (nosex == 482.758):
                            print(f'deltaX is {deltaX}, deltaY is {deltaY}, noseX is {nosex} so ear2X is {ear2X}')
                        ear2Y = nosey - 1/2*deltaX  

                        eye1 = [eye1X, eye1Y, 1]
                        eye2 = [eye2X, eye2Y, 1]
                        ear1 = [ear1X, ear1Y, 1]
                        ear2 = [ear2X, ear2Y, 1]

                        #shoulderDeltaX = rightShoulderX - leftShoulderX. If it's >0, left shoulder has lower x value,
                        #so left eye/ear will have lower x value
                        #Add new keypoints to playerKeypoints in the order: left_eye, right_eye, left_ear, right_ear
                        shoulderDeltaX = playerKeypoints[6] - playerKeypoints[15]
                        if (nosex == 482.758):
                            print(f'deltaX is {deltaX}, deltaY is {deltaY}, noseX is {nosex} so ear2X is {ear2X}')
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
                        #if ("2017-11-05-col-nyi-home12" in game_dir_name):
                        #    print(playerKeypoints)
                    else:
                        #If nose or neck is out of frame, just set all face keypoints to 0s
                        playerKeypoints.extend([0,0,1,0,0,1,0,0,1,0,0,1])

                    newPlayerKeypoints = []
                    for i in range(17):
                        cocoBodyPart = coco_body_parts.get(i)
                        existingIndex = existing_body_parts.get(cocoBodyPart)
                        newPlayerKeypoints.append(playerKeypoints[existingIndex * 3])
                        newPlayerKeypoints.append(playerKeypoints[existingIndex * 3 + 1])
                        newPlayerKeypoints.append(1)

                    player["keypoints"] = newPlayerKeypoints

                
                # Write the json to a file and save it
                outputFilenameString = f"{game_dir_full_path}/{game_dir_name}-17-keypoints.json"
                outputFile = open(outputFilenameString, "w")
                outputJson = json.dump(data, outputFile, indent=4)
                outputFile.close()