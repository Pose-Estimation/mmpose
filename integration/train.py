import pickle
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torchvision.models as models
import cv2
import matplotlib.ticker as mtick

from networkinte import IntegrationNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# train the network
def train_model(net, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        print("Training epoch %d" % (epoch + 1))

        running_loss = 0.0
        for i, data in enumerate(dataset_loader, 0):
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
                print("\t[Minibatches %d] loss: %.4f" % (i + 1, running_loss / 25))
                running_loss = 0.0


train_model(IntegrationNet, torch.nn.MSELoss, torch.optim.Adam, 100)


# # Splitting dataset
# train_size = int(0.8 * (len(data)))

# x_validation = data[train_size:]
# y_validation = new_labels[train_size:]
# x_train = data[:train_size]
# y_train = new_labels[:train_size]

# train_dataset = MyCustomDataset(
#     x_train,
#     y_train,
#     transforms.Compose(
#         [
#             transforms.ToTensor(),
#             transforms.Normalize(np.mean(x_train), np.std(x_train)),
#             torchvision.transforms.RandomAffine(
#                 5, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.1
#             ),
#         ]
#     ),
# )

# validation_dataset = MyCustomDataset(
#     x_validation,
#     y_validation,
#     transforms.Compose(
#         [
#             transforms.ToTensor(),
#             transforms.Normalize(np.mean(x_validation), np.std(x_validation)),
#         ]
#     ),
# )

# dataset_loader = torch.utils.data.DataLoader(
#     dataset=train_dataset, batch_size=128, shuffle=True
# )
# validation_loader = torch.utils.data.DataLoader(
#     dataset=validation_dataset, batch_size=128, shuffle=False
# )


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# # train the network
# def train_model(net, criterion, optimizer, num_epochs):
#     for epoch in range(num_epochs):  # loop over the dataset multiple times
#         print("Training epoch %d" % (epoch + 1))

#         running_loss = 0.0
#         for i, data in enumerate(dataset_loader, 0):
#             # get the inputs; data is a list of [inputs, labels]

#             inputs, labels = data[0].to(device), data[1].to(device)
#             # zero the parameter gradients
#             optimizer.zero_grad()

#             # forward + backward + optimize
#             outputs = net(inputs.float())

#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             # print statistics
#             running_loss += loss.item()
#             if i % 25 == 24:  # print every 100 mini-batches
#                 print("\t[Minibatches %d] loss: %.4f" % (i + 1, running_loss / 25))
#                 running_loss = 0.0


# # mnasnet0_5
# mnasnet0_5 = models.mnasnet0_5(num_classes=260, pretrained=False)
# mnasnet0_5.layers[0] = nn.Conv2d(
#     1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
# )

# train_model(
#     mnasnet0_5.to(device),
#     nn.CrossEntropyLoss(),
#     optim.Adam(mnasnet0_5.parameters(), lr=0.001, weight_decay=1e-5),
#     5,
# )
