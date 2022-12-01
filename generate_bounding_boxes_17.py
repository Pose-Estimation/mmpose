import math
import numpy as np
import json
import sys
import os

if __name__ == "__main__":
    PATH_TO_VIDEOPOSE = input("Enter the absolute path to your video_pose directory:")
    VIDEO_POSE_TYPES = {"No_penalty": 0, "Slashing": 1, "Tripping": 2}

    OS_DIR = ["train", "validate", "test"]

    annotation_id = 0  # increment this to have unique id for each annotation
    for video_dir_name in OS_DIR:
        print(f"Creating JSONs with 17 keypoints in {video_dir_name}:")
        file_path = f"{PATH_TO_VIDEOPOSE}/{video_dir_name}/{video_dir_name}-17-keypoints-coco.json"
        f = open(file_path)
        data = json.load(f)
        f.close()
        data_bboxonly = []

        #get image width and height, assuming they are the same size for all images in one video
        #Note: even now that it's images from multiple videos, it seems like all images in the dataset
        #are 640x360
        imageWidth = data["images"][0]["width"]
        imageHeight = data["images"][0]["height"]

        for player in data["annotations"] :
            playerKeypoints = player["keypoints"]
            minx = 99999
            maxx = -99999
            miny = 99999
            maxy = -99999

            outofrangekeypoints = []

            for i in range(17) :
                #if out of frame, skip this keypoint
                if playerKeypoints[i * 3] == 0 and playerKeypoints[i * 3 + 1] == 0 :
                    outofrangekeypoints.append(i)
                    continue

                #otherwise, check for a new minx, miny, maxx, maxy
                if playerKeypoints[i * 3] < minx :
                    minx = playerKeypoints[i * 3]
                if playerKeypoints[i * 3] > maxx : 
                    maxx = playerKeypoints[i * 3]
                if playerKeypoints[i * 3 + 1] < miny :
                    miny = playerKeypoints[i * 3 + 1]
                if playerKeypoints[i * 3 + 1] > maxy : 
                    maxy = playerKeypoints[i * 3 + 1]

            bbox = [0,0,0,0]

            if minx == 99999 or maxx == -99999 or miny == 99999 or maxy == -99999 :
                #player is fully out of frame, set bounding box to 0,0,0,0
                bbox = [0,0,0,0]
            else :
                # Box formed just around the keypoints will be too small; need to grow it
                # Currently growing it by double the face-neck distance (which for 17 keypoints is)
                # 4x the nose-ear distance
                nosex = playerKeypoints[0]
                nosey = playerKeypoints[1]
                noseVisibility = playerKeypoints[2]
                earx = playerKeypoints[3]
                eary = playerKeypoints[4]

                halfFaceDistance = 0
                # don't calculate half face distance if nose or ear is out of frame
                if (not ((nosex == 0 and nosey == 0) or (earx == 0 and eary == 0))):
                    halfFaceDistance = math.sqrt((nosex - earx) * (nosex - earx) + (nosey - eary) * (nosey - eary))

                    # Increment the size of the bounding box, making sure to stay in frame
                    minx = np.maximum(0.0, minx - 4*halfFaceDistance)
                    maxx = np.minimum(float(imageWidth), maxx + 4*halfFaceDistance)
                    miny = np.maximum(0.0, miny - 4*halfFaceDistance)
                    maxy = np.minimum(float(imageHeight), maxy + 4*halfFaceDistance)
                else:
                    #if nose or neck is out of frame, just increase by 20%
                    width = maxx - minx
                    height = maxy - miny
                    minx = np.maximum(0.0, minx -width * 0.1)
                    maxx = np.minimum(float(imageWidth), maxx + width * 0.1)
                    miny = np.maximum(0.0, miny - height * 0.1)
                    maxy = np.minimum(float(imageHeight), maxy + height * 0.1)

                #If there are any keypoints out of frame, check whether the player is close to the border and extend. 
                # 30 is chosen arbitrarily; better system for close would be ideal
                if len(outofrangekeypoints) > 0:
                    upperPoints = [0,1,2,3,4,5,6]
                    leftPoints = [5,7,9,11,13,15]
                    rightPoints = [6,8,10,12,14,16]
                    lowerPoints = [13,14,15,16]
                    #if nose, eyes, ears, or shoulders are out of frame, check if near top frame
                    if len(list(set(upperPoints) & set(outofrangekeypoints))) > 0 and miny < 30:
                        miny = 0
                    #if left side body parts are oof, check left side
                    if len(list(set(leftPoints) & set(outofrangekeypoints))) > 0 and minx < 30:
                        minx = 0
                    #if right side body parts are oof, check right side
                    if len(list(set(rightPoints) & set(outofrangekeypoints))) > 0 and imageWidth - maxx < 30:
                        maxx = imageWidth
                    #if knees or ankles are oof, check bottom side
                    if len(list(set(rightPoints) & set(outofrangekeypoints))) > 0 and imageHeight - maxy < 30:
                        maxy = imageWidth

                width = maxx - minx
                height = maxy - miny

                #bbox is format x,y,width,height, where x,y is the top left corner of the box
                bbox = [minx, miny, width, height]
            
            # Set the bbox attribute in the json data
            player["bbox"] = bbox

            player_bboxonly = {}
            player_bboxonly["bbox"] = bbox
            player_bboxonly["category_id"] = player["category_id"]
            player_bboxonly["image_id"] = player["image_id"]
            player_bboxonly["score"] = 1
            data_bboxonly.append(player_bboxonly)
        
        # Write the json to a file and save it
        outputFilenameString = f"{PATH_TO_VIDEOPOSE}/{video_dir_name}/{video_dir_name}-17-bbox-appended-coco.json"
        outputFile = open(outputFilenameString, "w")
        outputJson = json.dump(data, outputFile, indent=4)
        outputFile.close()

        outputFilenameString = f"{PATH_TO_VIDEOPOSE}/{video_dir_name}/{video_dir_name}-17-bbox-only-coco.json"
        outputFile = open(outputFilenameString, "w")
        outputJson = json.dump(data_bboxonly, outputFile, indent=4)
        outputFile.close()





