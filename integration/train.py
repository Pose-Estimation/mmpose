import torch
import numpy as np
import tqdm
from train_dataset import TrainInteDataset

from networkinte import IntegrationNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# train the network
def train_model(net, criterion, optimizer, num_epochs, dataset_loader):
    for epoch in tqdm(num_epochs):  # loop over the dataset multiple times
        print("Training epoch %d" % (epoch + 1))

        running_loss = 0.0
        for i, data in enumerate(dataset_loader):
            # get the inputs; data is a list of [inputs, labels]

            inputs, labels = data[0].to(device), data[1].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs.float())

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i % 25 == 24:  # print every 100 mini-batches
                print("\t[Minibatches %d] loss: %.4f" %
                      (i + 1, running_loss / 25))
                running_loss = 0.0


if __name__ == "__main__":
    PATH_TO_VIDEOPOSE = input(
        "Enter the absolute path to your video_pose/full_data directory:")
    training_file_path = f"{PATH_TO_VIDEOPOSE}/train/train-coco.json"
    loader = TrainInteDataset(training_file_path)
    train_model(IntegrationNet, torch.nn.MSELoss, torch.optim.Adam, 100, loader)