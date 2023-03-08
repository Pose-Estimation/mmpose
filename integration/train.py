import json
import time
import torch
from tqdm import tqdm
from train_dataset import TrainInteDataset
import TorchSUL.Model as M
from networkinte import IntegrationNet
import torch.optim.lr_scheduler as lr_scheduler
import wandb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# train the network
def train_model(net,
                criterion,
                optimizer,
                num_epochs,
                dataset_loader,
                validation_loader,
                valid_interval=99,
                patience=50):
    bar = tqdm(range(num_epochs))
    lr_scheduler.LinearLR(
        optimizer, 1, 0.5, int(num_epochs * 0.125), verbose=True)
    ms = time.time_ns() // 1000000
    log_file = open(f"./integration/logs{ms}.txt", "w")
    
    
    trigger = 0
    last_loss = float("inf")

    # wandb.watch(net, criterion)
    for epoch in bar:  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(dataset_loader):
            # get the inputs; data is a list of [inputs, labels]

            inputs, labels = torch.tensor(data[0]).to(device), torch.tensor(
                data[1]).to(device)
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
        bar.set_description("\nLoss: %.4f" % l)
        log_file.write("EPOCH: %d LOSS %.4f\n" % (epoch, l))

        if epoch % valid_interval == 0:
            net.eval()
            running_loss = 0
            with torch.no_grad():
                for data in validation_loader:
                    inputs, labels = torch.tensor(
                        data[0]).to(device), torch.tensor(data[1]).to(device)
                    outputs = net(inputs.float())

                    loss = criterion(outputs, labels)
                    running_loss += loss.item()
                    
            validation_loss = running_loss / len(validation_loader)
            validation_string = "VALIDATION LOSS: %.4f" % ( validation_loss)
            log_file.write(f"{validation_string}\n")
            print("\n" + validation_string)
            
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
    M.Saver(net).save(f'./integration/inte{num_epochs}.pth')


if __name__ == "__main__":
    PATH_TO_VIDEOPOSE = input(
        "Enter the absolute path to your video_pose/full_data directory:")

    # wandb.init(
    #     project="capstone", config={
    #         "learning_rate": 1e-4,
    #         "epochs": 800,
    #     })

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

    train_loader = TrainInteDataset(train_annotations)
    valid_loader = TrainInteDataset(validation_annotations)
    integration_net = IntegrationNet()
    pts_dumb = torch.zeros(2, 56)
    integration_net(pts_dumb)
    integration_net.to(device)
    train_model(integration_net, torch.nn.MSELoss(),
                torch.optim.Adam(integration_net.parameters(), lr=1e-3), 600,
                train_loader, valid_loader, 1)
    # wandb.finish()
