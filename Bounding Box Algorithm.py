import math
import numpy as np
import json
import sys

import cv2
import matplotlib.pyplot as plt
import matplotlib.image as im 

if __name__ == "__main__":
    #Get the file name from the first argument, load it into json
    folderAddress = ""
    if (len(sys.argv) > 0) :
        folderAddress = sys.argv[1]

    # Assumes a certain video structure
    # TODO: Currently hard-coded for no_penalty, will need to adjust this
    filenameString = "video_pose/No_penalty/" + folderAddress + "/" + folderAddress + "-coco.json"

    f = open(filenameString)
    data = json.load(f)
    f.close()

    #get image width and height, assuming they are the same size for all images in one video
    imageWidth = data["images"][0]["width"]
    imageHeight = data["images"][0]["height"]

    for player in data["annotations"] :
        playerNumberString = "p" + str(player["category_id"])
        playerKeypoints = player["keypoints"]
        minx = 99999
        maxx = -99999
        miny = 99999
        maxy = -99999

        # Currently ignoring the stick; should it be included in the box?
        for i in range(14) :
            #if out of frame, skip this keypoint
            if playerKeypoints[i * 3] == 0 and playerKeypoints[i * 3 + 1] == 0 :
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
            # Easy proposal: grow it on all sides by the half-face distance (distance from nose to neck)
            # This may need to be tinkered with to get more accurate boxes
            # Also: what to do when either is out of frame?
            nosex = playerKeypoints[0]
            nosey = playerKeypoints[1]
            neckx = playerKeypoints[3]
            necky = playerKeypoints[4]

            halfFaceDistance = 0
            # don't calculate half face distance if nose or neck is out of frame
            if (not ((nosex == 0 and nosey == 0) or (neckx == 0 and necky == 0))):
                halfFaceDistance = math.sqrt((nosex - neckx) * (nosex - neckx) + (nosey - necky) * (nosey - necky))

            # Increment the size of the bounding box, making sure to stay in frame
            minx = np.maximum(0.0, minx - halfFaceDistance)
            maxx = np.minimum(float(imageWidth), maxx + halfFaceDistance)
            miny = np.maximum(0.0, miny - halfFaceDistance)
            maxy = np.minimum(float(imageHeight), maxy + halfFaceDistance)

            width = maxx - minx
            height = maxy - miny

            #bbox is format x,y,width,height, where x,y is the top left corner of the box
            bbox = [minx, miny, width, height]
        
        # Set the bbox attribute in the json data
        player["bbox"] = bbox
    
    # Write the json to a file and save it
    outputFilenameString = "video_pose/No_penalty/" + folderAddress + "/" + folderAddress + "-bbox.json"
    outputFile = open(outputFilenameString, "w")
    outputJson = json.dump(data, outputFile, indent=4)
    outputFile.close()





