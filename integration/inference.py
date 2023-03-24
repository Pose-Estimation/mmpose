import json
import time
import torch
from tqdm import tqdm
from train_dataset import TrainInteDataset
import TorchSUL.Model as M
from networkinte import IntegrationNet
import torch.optim.lr_scheduler as lr_scheduler
import wandb
import inteutil
from matching import posematcher

# Loading from internet TODO: show viet this
integration_net = IntegrationNet()
state_dict = torch.load(pre_trained_path)
integration_net.load_state_dict(state_dict)
custom.to(device)

print(f"model {pre_trained_path} loaded")

# TODO: Show this is how we evluate
with torch.no_grad():
    model.eval()
    result = model(input_image)


# Predict with td, bu

"""
- train integration / top  down/ bottom up individually
- get prediction from top down / bottom up
- predictions to integration net
- matching the prediction

"""


def test_accuracy(dataset_loader, net):
    correct = 0
    total = 0
    net.eval()
    with torch.no_grad():
        for data in dataset_loader:
            images, labels = data[0].to(torch.device), data[1].to(torch.device)
            outputs = net(images.float())
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


if __name__ == "__main__":
    PATH_TO_TD = input("Enter the absolute path to your top-down results json")
    PATH_TO_TD = "D:\DocumentsD\Captsone\keypoints\td_keypoints.json"

    PATH_TO_BU = input("Enter the absolute path to your bottom-up results json")
    PATH_TO_BU = "D:\DocumentsD\Captsone\keypoints\bu_keypoints.json"

    ## match the poses
    print("Matching poses from two branches...")
    matcher = posematcher.PoseMatcher(
        top_down_path=PATH_TO_TD,
        btm_up_path=PATH_TO_BU,
    )
    matcher.match(pts_out_path="./integration/pred_bu/")

    # infer the integrated results
    print("Inferring the integrated poses...")

    # data loader
    data = inteutil.InteDataset(
        bu_path="./mupots/pred_bu/", td_path="./mupots/pred/"
    )  # TODO: ASK VIET: TrainInteDataset OR InteDataset ??? How we doin that
    # initialize the network

    net = IntegrationNet()
    pts_dumb = torch.zeros(2, 84)
    net(pts_dumb)

    PATH_TO_TD = input("Enter the absolute path to your integration network checkpoint")
    PATH_TO_TD = "D:\DocumentsD\Captsone\keypoints\inte.pth"
    M.Saver(net).restore(PATH_TO_TD)  # TODO: show viet this is how we load
    net.cuda()

    # create paths
    if not os.path.exists("./mupots/pred_inte/"):
        os.makedirs("./mupots/pred_inte/")

    with torch.no_grad():
        all_pts = defaultdict(list)
        for src_pts, src_dep, vid_inst in tqdm(data):
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
