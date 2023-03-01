# Imports
import torch
import torchvision
import torch.nn as nn                           # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim                     # For all Optimization algorithms, SGD, Adam, etc.
import torch.nn.functional as F                 # All functions that don't have any parameters
from torch.utils.data import DataLoader         # Gives easier dataset managment and creates mini batches
import torchvision.datasets as datasets         # Has standard datasets we can import in a nice way
import torchvision.transforms as transforms     # Transformations we can perform on our dataset


def check_accuracy(loader, model):
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

        print(
            # f'{mode} : accuracy = {float(num_correct)/float(num_samples)*100:.2f}'
            f"{mode} : {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}"
        )

    model.train()


# Create Fully Connected Network
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)  # flatten data : 28x28 -> 784,  x = x.view(-1,784)
        x = F.softmax(self.fc1(x))
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

    # Initialize nerual network
    model = NN(input_size=784, num_classes=10).to(device)

    # define Loss and optimizer
    loss_criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.05)

    # Train model
    for epoch in range(num_epochs):
        for batch_idx, (input_image, targets) in enumerate(train_loader):

            # Get input_image to CPU/cuda
            input_image = input_image.to(device=device)
            targets = torch.nn.functional.one_hot(targets, num_classes=10)
            targets = targets.to(torch.float32)
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
        check_accuracy(train_loader, model)
        check_accuracy(test_loader, model)

x = np.arange(1, 10, 0.1)
y1 = float(train_loader)/float(model)*100
y2 = float(test_loader)/float(model)*100
plt.plot(x, y1, label = 'train')
plt.plot(x, y2, linestyle = "--", label = "test")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.title('accuracy')
plt.legend()

plt.show()
