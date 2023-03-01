# Imports
import torch
import torchvision
import torch.nn as nn                           # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim                     # For all Optimization algorithms, SGD, Adam, etc.
import torch.nn.functional as F                 # All functions that don't have any parameters
from torch.utils.data import DataLoader         # Gives easier dataset managment and creates mini batches
import torchvision.datasets as datasets         # Has standard datasets we can import in a nice way
import torchvision.transforms as transforms     # Transformations we can perform on our dataset

import matplotlib.pyplot as plt
import numpy as np


def check_accuracy(loader, model, criterion):
    if loader.dataset.train:
        mode = 'train'
    else:
        mode = 'test '

    model.eval()

    num_correct = 0
    num_samples = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            predictions = model(x)  # tensor([[ 1.2100e-01, -3.0610e-02,  1.3518e-01, -6.9840e-02,  3.5525e-01, 1.2030e-02, -8.2505e-02,  1.8832e-01, -7.2998e-02,  1.0412e-01],
            probs, predictions = predictions.max(1)  # refer torch.max -> probs = tensor([0.35525,...]),  predictions = tensor([4,...])
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

            accuracy = float(num_correct)/float(num_samples)*100

            # get loss
            predictions = model(x)
            loss = criterion(predictions, y)
            loss = loss.item()  # tensor -> value

        print(
            # f'{mode} : accuracy = {float(num_correct)/float(num_samples)*100:.2f}'
            f"{mode} : {num_correct} / {num_samples} with accuracy {accuracy:.2f}"
        )

    model.train()

    return accuracy, loss


def draw_graph(train_accuracy_list, train_loss_list, test_accuracy_list, test_loss_list):
    # draw graph : accuracy
    x = np.arange(len(train_accuracy_list))
    plt.figure(1)
    plt.plot(x, train_accuracy_list, label='train', markevery=1)
    plt.plot(x, test_accuracy_list, label='test', markevery=1)
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.ylim(0, 100)
    plt.legend(loc='lower right')
    # plt.show()

    # draw graph : loss
    # x = np.arange(len(train_loss_list))
    plt.figure(2)
    plt.plot(x, train_loss_list, label='train', markevery=1)
    plt.plot(x, test_loss_list, label='test', markevery=1)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend(loc='upper right')
    plt.show()


# define CNN
class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=6,
            kernel_size=(6, 6),
            stride=(1, 1),
            padding=(3, 3),
        )
        self.conv2 = nn.Conv2d(
            in_channels=6,
            out_channels=12,
            kernel_size=(5, 5),
            stride=(2, 2),
            padding=(1, 1),   # padding=(1, 1) : 1 epoch 94.01, padding=(2, 2) : 1 epoch 92.25
        )
        self.conv3 = nn.Conv2d(
            in_channels=12,
            out_channels=24,
            kernel_size=(4, 4),
            stride=(2, 2),
            padding=(1, 1),
        )
        self.fc1 = nn.Linear(24 * 7 * 7, 200)
        self.fc2 = nn.Linear(200, num_classes)

        self.dropout = nn.Dropout(p=0.25)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.dropout(x)

        x = F.relu(self.conv2(x))
        x = self.dropout(x)

        x = F.relu(self.conv3(x))
        x = self.dropout(x)

        x = x.reshape(x.shape[0], -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        x = F.softmax(self.fc2(x))

        return x
#dropout은 마지막 relu에서 한번만 해주면 됨 위의 relu마다 dropout을 해주면 ㅋ의미도 없거니와 엄청난 컴퓨팅파워를 잡아먹는다

if __name__ == '__main__':

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    # learning_rate = 0.001
    # batch_size = 100
    in_channel = 1
    num_classes = 10
    num_epochs = 20

    # Load Data
    train_dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
    test_dataset = datasets.MNIST(root="dataset/", train=False, transform=transforms.ToTensor(), download=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=100, shuffle=True)

    # create neural network
    model = CNN(in_channel, num_classes).to(device)

    # Loss and optimizer
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train model
    train_accuracy_list = []
    train_loss_list = []
    test_accuracy_list = []
    test_loss_list = []
    for epoch in range(num_epochs):
        for batch_idx, (input_image, targets) in enumerate(train_loader):

            # Get input_image to CPU/cuda
            input_image = input_image.to(device=device)
            targets = targets.to(device=device)

            # predict(forward)
            predictions = model(input_image)

            # loss 계산
            loss = loss_criterion(predictions, targets)

            # update weight(SGD)
            optimizer.zero_grad()
            loss.backward()    # gradient 계산
            optimizer.step()   # update weight

        print(f'epoch = {epoch} : loss = {loss:.3f}')

        # Check accuracy on training & test to see how good our model
        train_accuracy, train_loss = check_accuracy(train_loader, model, loss_criterion)
        train_accuracy_list.append(train_accuracy)
        train_loss_list.append(train_loss)

        test_accuracy, test_loss = check_accuracy(test_loader, model, loss_criterion)
        test_accuracy_list.append(test_accuracy)
        test_loss_list.append(test_loss)

        print(f'epoch = {epoch} : train_loss = {train_loss:.4f}, test_loss = {test_loss:.4f},')

    # accuracy/loss graph
    draw_graph(train_accuracy_list, train_loss_list, test_accuracy_list, test_loss_list)



