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
            x = x.reshape(x.shape[0], -1)

            predictions = model(x)  # tensor([[ 1.2100e-01, -3.0610e-02,  1.3518e-01, -6.9840e-02,  3.5525e-01, 1.2030e-02, -8.2505e-02,  1.8832e-01, -7.2998e-02,  1.0412e-01],
            probs, predictions = predictions.max(1)  # refer torch.max -> probs = tensor([0.35525,...]),  predictions = tensor([4,...])
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

            accuracy = float(num_correct)/float(num_samples)*100

            # get loss
            predictions = model(x)
            # y = torch.nn.functional.one_hot(y, num_classes=10)
            # y = y.to(torch.float32)
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


# Define Fully Connected Network
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 60)
        self.fc4 = nn.Linear(60, 30)
        self.fc5 = nn.Linear(30, num_classes)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)   # flatten data : 28x28 -> 784,  x = x.view(-1,784)
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        x = F.sigmoid(self.fc4(x))
        x = F.softmax(self.fc5(x))
        return x


if __name__ == '__main__':

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    # input_size = 784
    # num_classes = 10
    # learning_rate = 0.05
    # batch_size = 100
    num_epochs = 10

    # Load Data
    train_dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
    test_dataset = datasets.MNIST(root="dataset/", train=False, transform=transforms.ToTensor(), download=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=100, shuffle=True)

    # create nerual network
    model = NN(input_size=784, num_classes=10).to(device)

    # define Loss and optimizer
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002)

    # Train model
    train_accuracy_list = []
    train_loss_list = []
    test_accuracy_list = []
    test_loss_list = []
    for epoch in range(num_epochs):
        for batch_idx, (input_image, targets) in enumerate(train_loader):

            # Get input_image to CPU/cuda
            input_image = input_image.to(device=device)
            # targets = torch.nn.functional.one_hot(targets, num_classes=10)
            # targets = targets.to(torch.float32)
            targets = targets.to(device=device)

            # predict(forward)
            predictions = model(input_image)

            # loss ??????
            loss = loss_criterion(predictions, targets)

            # update weight(SGD)
            optimizer.zero_grad()
            loss.backward()    # gradient ??????
            optimizer.step()   # update weight

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



