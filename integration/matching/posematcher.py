from collections import defaultdict
import json
import pickle
import numpy as np
import os
from matching.norm_pose import procrustes
from tqdm import tqdm
from matching.utils import format_keypoints


class PoseMatcher:

    def __init__(self, top_down_path, btm_up_path):
        self.top_down_path = top_down_path
        self.btm_up_path = btm_up_path

    def _best_match(self, ref, targets):

        def OKS(p1, p2):
            sigma = 25
            # selected_idx = np.int64([14, 8,9, 11,12, 5,6, 2,3])
            p1_selected = p1[:2]
            p2_selected = p2[:2]
            # p1_selected -= p1_selected[:,0:1]
            # p2_selected -= p2_selected[:,0:1]
            dist = np.square(p1_selected - p2_selected).sum(axis=0)
            result = np.exp(-dist / (2 * sigma**2)).mean()
            return result

        max_idx = 0
        max_oks = 0
        max_pts = 0
        for i in range(len(targets)):
            aligned_target = procrustes(targets[i], ref)
            oks = OKS(ref, aligned_target)
            if oks > max_oks:
                max_oks = oks
                max_idx = i
                max_pts = aligned_target
        return max_pts, max_idx

    def match(self, pts_out_path):
        # create directory
        if not os.path.exists(pts_out_path):
            os.makedirs(pts_out_path)

        # Load predictions
        bu_estimations = {}
        bu_predictions = json.load(open(self.btm_up_path))
        for pred in bu_predictions:
            image_id = pred["image_id"]
            if image_id in bu_estimations:
                bu_estimations[image_id].append(pred)
            else:
                bu_estimations[image_id] = [pred]

        # we sort here so it's in correct order, to prevent some linux os produce incorrect order
        td_estimations = json.load(open(self.top_down_path))
        prev = None
        results = []
        results_export = []

        for t in tqdm(td_estimations):
            img_id = t["image_id"]

            td_pts = format_keypoints(t["keypoints"])
            # match
            bu_pred = bu_estimations[img_id]
            bu_pred = np.array(
                [format_keypoints(pred["keypoints"]) for pred in bu_pred])
            p_aligned, _ = self._best_match(td_pts, bu_pred)
            if prev is None or not prev == img_id:
                results_export.append(results)
                results.clear()
            results.append(p_aligned)

            prev = img_id
        return results_export
