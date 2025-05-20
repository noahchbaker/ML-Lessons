#This one is gonna use class and idx2numpy, Im gonna try the card one later with the dir Image folder stuff
# want to use pygui to create a peice of software to write your own numbers

import numpy as np
import pandas as pd
import torch
from torch import nn 
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import idx2numpy

from torchvision.transforms import transforms


transform = transforms.Compose([
    transforms.RandomRotation(degrees=20), # Rotate by 15 degrees
    transforms.RandomResizedCrop(28, scale=(0.4, 1)), # Crop to 28x28
    transforms.Resize((28,28)),
    transforms.Normalize((0.1307,), (0.3081,)) # Normalize
])


class MNISTDataset (Dataset):
    def __init__(self, imagearray, labels, transform = None):
        self.imgSet = torch.tensor(imagearray, dtype = torch.float32).unsqueeze(1) / 255.0
        self.labels = torch.tensor(labels)
        self.transform = transform

        #or self.imgSet = ImageFolder(root = dir, transform = transform)

    def __len__(self):
        return(len(self.imgSet))

    def __getitem__(self, index):
        image = self.imgSet[index]
        if self.transform:
            image = self.transform(image)
        return image, self.labels[index]
    

imagefiletrain = '/Users/noahbaker/ML-Lessons/Day2-MNIST/MNIST/train-images.idx3-ubyte'
imagearraytrain = idx2numpy.convert_from_file(imagefiletrain)

imagefiletest = '/Users/noahbaker/ML-Lessons/Day2-MNIST/MNIST/t10k-images.idx3-ubyte'
imagearraytest = idx2numpy.convert_from_file(imagefiletest)

imagefiletrainlabel = '/Users/noahbaker/ML-Lessons/Day2-MNIST/MNIST/train-labels.idx1-ubyte'
imagearraytrainlabel = idx2numpy.convert_from_file(imagefiletrainlabel)

imagefiletestlabel = '/Users/noahbaker/ML-Lessons/Day2-MNIST/MNIST/t10k-labels.idx1-ubyte'
imagearraytestlabel = idx2numpy.convert_from_file(imagefiletestlabel)


print(f"image array train: {imagearraytrain.shape}")
print(f"image array test: {imagearraytest.shape}")
print(f"image array train label: {imagearraytrainlabel.shape}")
print(f"image array test label: {imagearraytestlabel.shape}")


#plt.imshow(imagearray[4], cmap=plt.cm.binary)
#plt.show()

training_data = MNISTDataset(imagearraytrain, imagearraytrainlabel, transform=transform)
test_data = MNISTDataset(imagearraytest, imagearraytestlabel, transform=transform)



'''print(data[4])
print(len(data))'''

class NeuralNetwork(nn.Module):
    def __init__ (self):
        super().__init__()
        self.flatten = nn.Flatten() #maybe add back later
        self.SequenceLayer = nn.Sequential (
            nn.Linear(28*28,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,10),
        )
         
    def forward(self, x):
        x = self.flatten(x)
        return self.SequenceLayer(x)


andy = NeuralNetwork()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(andy.parameters(), lr = 5e-5)


def train(Dataloader, model, loss_fn, optimizer):
    model.train()
    train_loss, correct, total_samples = 0, 0, 0
    for batch, (number, label) in enumerate(Dataloader):
        pred = model(number)
        loss = loss_fn(pred, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        correct += ((pred.argmax(1) == label).type(torch.float).sum().item())
        train_loss += loss.item() * number.size(0)
        total_samples += number.size(0)      

        accuracy = correct/total_samples
        avg_loss = train_loss/total_samples

        if(batch % 100 == 0):
            print(f"for batch: {batch} \t correct% = {100* accuracy} \t avg loss: {avg_loss}")


def test(Dataloader, model, loss_fn):
    model.eval()
    test_loss, correct, total_samples = 0,0,0
    with torch.no_grad():
        for number, label in Dataloader:
            pred = model(number)
            loss = loss_fn(pred, label)

            test_loss += loss.item() * number.size(0)
            total_samples += number.size(0)  

            correct += ((pred.argmax(1) == label).type(torch.float).sum().item())

        accuracy = correct/total_samples
        avg_loss = test_loss/total_samples

        print(f"TEST:::  correct% = {100 * accuracy} \t avg loss: {avg_loss}")


batch_size = 32
train_dataloader = DataLoader(training_data, batch_size = batch_size)
test_dataloader = DataLoader(test_data, batch_size = batch_size)


'''images, labels = next(iter(train_dataloader))
plt.imshow(images[0].squeeze(), cmap="gray")
plt.title(f"Label: {labels[0]}")
plt.show()'''


#print(next(iter(train_dataloader)))

epochs = 10
for t in range (epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, andy, loss_fn, optimizer)
    test(test_dataloader, andy, loss_fn)


torch.save(andy, 'number_model_newTransform.pth')