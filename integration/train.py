import torch
import numpy as np
from tqdm import tqdm
from train_dataset import TrainInteDataset
import TorchSUL.Model as M
from networkinte import IntegrationNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# train the network
def train_model(net, criterion, optimizer, num_epochs, dataset_loader):
    bar = tqdm(range(num_epochs))
    for _ in bar:  # loop over the dataset multiple times
        # print("Training epoch %d" % (epoch + 1))
        running_loss = 0.0
        for i , data in enumerate(dataset_loader):
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

        bar.set_description("\nLoss: %.4f" % (running_loss / len(dataset_loader)))
    M.Saver(net).save(f'./integration/inte{num_epochs}.pth')


if __name__ == "__main__":
    PATH_TO_VIDEOPOSE = input(
        "Enter the absolute path to your video_pose/full_data directory:")
    training_file_path = f"{PATH_TO_VIDEOPOSE}/train/train-coco.json"
    loader = TrainInteDataset(training_file_path, 32)
    integration_net = IntegrationNet()
    pts_dumb = torch.zeros(2, 84)
    integration_net(pts_dumb)
    integration_net.to(device)
    train_model(
        integration_net,
        torch.nn.MSELoss(),
        torch.optim.Adam(integration_net.parameters(), lr=1e-4),
        200,
        loader,
    )
