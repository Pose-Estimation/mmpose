##NOTE: This script requires MMPose version 1.0 (branch 1.x) to run to have access to script demo/image_demo.py.
##It cannot be run on this branch of the project.

from PIL import Image
import os
import subprocess

PATH_TO_VIDEOPOSE = input("Enter the absolute path to your video_pose directory:")
#sample config file: configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py
#CONFIG_FILE = input("Enter the absolute path to the config file you would like to use:")
#sample checkpoint link: https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth
#CHECKPOINT_LINK = input("Enter the checkpoint link:")

CONFIG_FILE = "configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py"
CHECKPOINT_LINK = "https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth"
base_dir_path = f"{PATH_TO_VIDEOPOSE}/train_heatmaps"

subdirectories = [f.path for f in os.scandir(base_dir_path) if f.is_dir()]
subdirectory_names = [os.path.basename(d) for d in subdirectories]
subdirectory_names.sort()

for directory in subdirectory_names :
    bash_command = f"python demo/image_demo.py {PATH_TO_VIDEOPOSE}/train_heatmaps/{directory}/{directory}.png {CONFIG_FILE} {CHECKPOINT_LINK} --out-file {PATH_TO_VIDEOPOSE}/train_heatmaps/{directory}/{directory}-combined.png --draw-heatmap"
    print(f"bash_command is {bash_command}")
    process = subprocess.run(bash_command, shell=True, text=True)
    #TODO : will this lead to redownloading the checkpoint file every time? If so, will need to find a more efficient way

    image = Image.open(f"{PATH_TO_VIDEOPOSE}/train_heatmaps/{directory}/{directory}-combined.png")
    width, height = image.size
    heatmap_image = image.crop((0, height/2, width, height))
    heatmap_image.save(f"{PATH_TO_VIDEOPOSE}/train_heatmaps/{directory}/{directory}-heatmap.png")