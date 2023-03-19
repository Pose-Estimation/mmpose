import json
import os
import uuid
import torch
from tqdm import tqdm
from train_dataset import TrainInteDataset
import TorchSUL.Model as M
from networkinte import IntegrationNet
import torch.optim.lr_scheduler as lr_scheduler
import wandb
from mmpose.core.evaluation.top_down_eval import keypoint_pck_accuracy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# train the network
def train_model(path,
                net,
                criterion,
                optimizer,
                num_epochs,
                dataset_loader,
                validation_loader,
                valid_interval=99,
                patience=50):
    bar = tqdm(
        range(num_epochs), leave=False, bar_format='{l_bar}{bar:10}{r_bar}')

    lr_scheduler.LinearLR(
        optimizer, 1, 0.5, int(num_epochs * 0.125), verbose=True)
    log_file = open(f"{path}/logs.txt", "w")

    trigger = 0
    last_loss = float("inf")

    # wandb.watch(net, criterion)
    for epoch in bar:  # loop over the dataset multiple times

        running_loss = 0.0
        for _, data in enumerate(dataset_loader):
            # get the inputs; data is a list of [inputs, labels]

            inputs, labels, _ = data
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs.float())

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

        l = running_loss / len(dataset_loader)
        bar.set_description(f'Loss: {l:.4}')
        log_file.write("EPOCH: %d LOSS %.6f\n" % (epoch, l))

        if epoch % valid_interval == 0:
            net.eval()
            running_loss = 0
            pck_running_avg = 0

            with torch.no_grad():
                for data in validation_loader:
                    inputs, labels, masks = data
                    outputs = net(inputs.float())

                    loss = criterion(outputs, labels)
                    running_loss += loss.item()

            validation_loss = running_loss / len(validation_loader)
            validation_string = "VALIDATION LOSS: %.6f" % (validation_loss)
            log_file.write(f"{validation_string}\n")
            print(f"\nVALIDATION LOSS {validation_loss:.4} \n")

            if validation_loss > last_loss:
                trigger += 1

                if trigger == patience:
                    print(f"Early Stopping Trigger with patience: {patience}")
                    break
            else:
                trigger = 0

            last_loss = validation_loss

            net.train()

    log_file.close()
    M.Saver(net).save(f'{path}/inte.pth')


if __name__ == "__main__":
    PATH_TO_VIDEOPOSE = 'C:/Users/Admin/Desktop/Datasets/hockey_dataset/full_data'

    # wandb.init(
    #     project="capstone", config={
    #         "learning_rate": 1e-4,
    #         "epochs": 800,
    #     })

    CONFIG = {"lr": 1e-4, "epochs": 800, "batch_size": 32, "weight_decay":0}
    path = f"./integration/results/{uuid.uuid4()}"
    os.mkdir(path)
    config_f = open(f'{path}/config.txt', 'w')
    json.dump(CONFIG, config_f)
    config_f.close()

    #Getting annotations
    training_file_path = f"{PATH_TO_VIDEOPOSE}/train/train-coco.json"
    f = open(training_file_path)
    data = json.load(f)
    f.close()
    train_annotations = data["annotations"]

    validation_file_path = f"{PATH_TO_VIDEOPOSE}/validate/validate-coco.json"
    f = open(validation_file_path)
    data_valid = json.load(f)
    f.close()

    validation_annotations = data_valid["annotations"]

    print("LOADING DATASETS")
    train_loader = TrainInteDataset(train_annotations, CONFIG["batch_size"])
    valid_loader = TrainInteDataset(validation_annotations,
                                    CONFIG["batch_size"])
    print("DONE")
    print("-" * 100)

    integration_net = IntegrationNet()
    pts_dumb = torch.zeros(2, 56)
    integration_net(pts_dumb)
    integration_net.to(device)
    train_model(
        path, integration_net, torch.nn.MSELoss(),
        torch.optim.Adam(
            integration_net.parameters(),
            lr=CONFIG["lr"],
            weight_decay=CONFIG["weight_decay"]), CONFIG["epochs"],
        train_loader, valid_loader, 1)
    # wandb.finish()
