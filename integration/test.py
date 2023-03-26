from collections import defaultdict
import json
import os
import pickle
import numpy as np
import torch
from tqdm import tqdm
from test_dataset import TestInteDataset
import TorchSUL.Model as M
from networkinte import IntegrationNet
from matching import posematcher

if __name__ == "__main__":
    PATH_TO_TOP_DOWN = input("Enter the absolute path to your top-down results json")
    PATH_TO_TOP_DOWN = "D:\DocumentsD\Captsone\keypoints\\td_keypoints.json"

    PATH_TO_BOTTUM_UP = input("Enter the absolute path to your bottom-up results json")
    PATH_TO_BOTTUM_UP = "D:\DocumentsD\Captsone\keypoints\\bu_keypoints.json"

    # Match the poses
    print("Matching poses from two branches...")
    matcher = posematcher.PoseMatcher(
        top_down_path=PATH_TO_TOP_DOWN,
        btm_up_path=PATH_TO_BOTTUM_UP,
    )
    bottom_up_kpts = matcher.match(pts_out_path="./integration/pred_bu/")

    # Getting td annotations
    f = open(PATH_TO_TOP_DOWN)
    top_down_data = json.load(f)
    f.close()
    top_down_annots = top_down_data["annotations"]

    # Getting ground-truth annotations
    PATH_TO_GROUND_TRUTH = input(
        "Enter the absolute path to your ground-truth measurements json"
    )
    PATH_TO_GROUND_TRUTH = (
        "D:\DocumentsD\Captsone\\video_pose\\full_data\\test\\test-bbox-appended.json"
    )
    f = open(PATH_TO_GROUND_TRUTH)
    ground_truth_data = json.load(f)
    f.close()
    ground_truth_annots = ground_truth_data["annotations"]

    test_loader = TestInteDataset(
        ground_truth_annots, bottom_up_kpts, top_down_annots, batch_size=32
    )

    # Initialize/load the network
    net = IntegrationNet()
    pts_dumb = torch.zeros(2, 84)
    net(pts_dumb)

    PATH_TO_TD = input("Enter the absolute path to your integration network checkpoint")
    PATH_TO_TD = "D:\DocumentsD\Captsone\keypoints\inte.pth"
    M.Saver(net).restore(PATH_TO_TD)
    net.cuda()

    # create paths
    if not os.path.exists("/pred_inte/"):
        os.makedirs("/pred_inte/")

    with torch.no_grad():
        # TODO: theres no eval in test_integrate or in the source repo when they test the integration network? (https://github.com/3dpose/3D-Multi-Person-Pose/blob/main/calculate_mupots_integrate.py)
        #     model.eval()
        #     result = model(input_image)

        all_pts = defaultdict(list)
        for src_pts, src_dep, vid_inst in tqdm(test_loader):
            src_pts = torch.from_numpy(src_pts).cuda()
            res_pts = net(src_pts)
            res_pts = res_pts.cpu().numpy()

            # save results
            # TODO change to coco format??
            i, j = vid_inst
            all_pts[i].insert(j, res_pts)

        for k in all_pts:
            result = np.stack(all_pts[k], axis=1)
            pickle.dump(result, open("./mupots/pred_inte/%d.pkl" % (k + 1), "wb"))
