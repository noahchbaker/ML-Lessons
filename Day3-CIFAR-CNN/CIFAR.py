import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

from progress_bar import InitBar

'''
import torchvision.models as models

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 10)  # CIFAR-10 has 10 classes

'''

labels = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4), #CIFAR has 32x32 images
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
])


'''
Input: 32, height 32, and depth 3 (32x32x3)
'''

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=32, shuffle=True)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)

testloader = torch.utils.data.DataLoader(
    trainset, batch_size=32, shuffle=True)


model = CNN()

optimizer = optim.Adam(model.parameters(), lr=1e-3) #momentum and weight decay
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
loss_fn = nn.CrossEntropyLoss()


def train(model, dataloader, epoch):
    model.train()
    pbar = InitBar()  # initialize progress bar
    running_loss = 0.0
    
    for batch, (img, label) in enumerate(dataloader):
        optimizer.zero_grad()
        pred = model(img)
        loss = loss_fn(pred, label)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch % 500 == 0:    
            print(f'[{epoch + 1}, {batch + 1:5d}] loss: {running_loss / 2000:.3f}')
            loss = 0.0

        percent = int(100 * (batch + 1) / 1563)
        pbar(percent)  # update progress bar


correct_pred = {classname: 0 for classname in labels}
total_pred = {classname: 0 for classname in labels}


def test(model, dataloader):
    correct, loss = 0,0
    model.eval()
    for (img, label) in dataloader:
        with torch.no_grad():
            prediction = model(img)
            _, output = torch.max(prediction, 1)
            for true_label, predicted_label in zip(label, output):
                class_name = labels[true_label]
                total_pred[class_name] += 1
                if predicted_label == true_label:
                    correct_pred[class_name] += 1
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')


epochs = 25
for t in range (epochs):
    train(model, trainloader, t)
    test(model, testloader)


PATH = './cifar_net.pth'
torch.save(model.state_dict(), PATH)

        




