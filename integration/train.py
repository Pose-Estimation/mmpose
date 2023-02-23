import json
import torch
import random
from tqdm import tqdm
from train_dataset import TrainInteDataset
import TorchSUL.Model as M
from networkinte import IntegrationNet
import torch.optim.lr_scheduler as lr_scheduler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# train the network
def train_model(net, criterion, optimizer, num_epochs, dataset_loader, validation_loader, valid_interval=100):
    bar = tqdm(range(num_epochs))
    lr_scheduler.LinearLR(optimizer, 1, 0.1, int(num_epochs * 0.8))
    
    for epoch in bar:  # loop over the dataset multiple times
        # print("Training epoch %d" % (epoch + 1))
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

        bar.set_description("\nLoss: %.4f" %
                            (running_loss / len(dataset_loader)))
        
        if epoch and epoch%valid_interval==0:
            net.eval()
            running_loss = 0 
            with torch.no_grad():
                for data in validation_loader:
                    inputs, labels = torch.tensor(data[0]).to(device), torch.tensor(
                    data[1]).to(device)
                    outputs = net(inputs.float())

                    loss = criterion(outputs, labels)
                    running_loss+=loss.item()
            
            print("VALIDATION LOSS: %.4f" %(running_loss / len(validation_loader))) 
            net.train()
                
                
        
    M.Saver(net).save(f'./integration/inte{num_epochs}.pth')


if __name__ == "__main__":
    PATH_TO_VIDEOPOSE = input(
        "Enter the absolute path to your video_pose/full_data directory:")
    
    #Getting annotations
    training_file_path = f"{PATH_TO_VIDEOPOSE}/train/train-coco.json"
    f = open(training_file_path)
    data = json.load(f)
    f.close()
    annotations = data["annotations"]
    random.shuffle(annotations)
    
    split = int(len(annotations) * 0.80)
    
    train_loader = TrainInteDataset(annotations[:split])
    valid_loader = TrainInteDataset(annotations[split:])
    integration_net = IntegrationNet()
    pts_dumb = torch.zeros(2, 84)
    integration_net(pts_dumb)
    integration_net.to(device)
    train_model(
        integration_net,
        torch.nn.MSELoss(),
        torch.optim.Adam(integration_net.parameters(), lr=5e-4),
        800,
        train_loader,
        valid_loader
    )
