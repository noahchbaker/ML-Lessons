from CreateNP import createCluster
import numpy as np
import pandas as pd
import torch
from torch import nn 
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

centers = np.array([[2,5], [10,16], [4,2], [12,12], [20,20], [12, 3], [22,10]])
X, y = createCluster(centers, 0.8, 100, 2)
dataset = pd.DataFrame({'X': X[:, 0], 'Y': X[:, 1], 'Target': y})
#print(dataset)

train_df, test_df = train_test_split(dataset, test_size=0.2, random_state=42) # 80% train, 20% test

class CoupleClusters(Dataset):
    def __init__ (self, dataframe):
        self.dataframe = dataframe
        self.X = torch.tensor(dataframe.drop(columns = 'Target').values, dtype=torch.float32)
        self.y = torch.tensor(dataframe['Target'].values, dtype = torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

train_dataset = CoupleClusters(train_df)
test_dataset = CoupleClusters(test_df)

#print(len(train_dataset))
#print(train_dataset[100])

class NeuralNetwork(nn.Module):
    def __init__ (self):
        super().__init__()
        #self.flatten = nn.Flatten() #maybe add back later
        self.mapping = nn.Sequential (
            nn.Linear(2,10),
            nn.ReLU(),
            nn.Linear(10,10),
            nn.ReLU(),
            nn.Linear(10,7),
        )
         
    def forward(self, x):
        return self.mapping(x)
    

andy = NeuralNetwork()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(andy.parameters(), lr = 1e-1)


def train(Dataloader, model, loss_fn, optimizer):
    model.train()
    train_loss, correct, total_samples = 0, 0, 0
    for batch, (X, y) in enumerate(Dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        correct += ((pred.argmax(1) == y).type(torch.float).sum().item())
        train_loss += loss.item() * X.size(0)
        total_samples += X.size(0)      

        accuracy = correct/total_samples
        avg_loss = train_loss/total_samples

        if(batch % 5 == 0):
            print(f"for batch: {batch} \t correct% = {100* accuracy} \t avg loss: {avg_loss}")


def test(Dataloader, model, loss_fn):
    model.eval()
    test_loss, correct, total_samples = 0,0,0
    with torch.no_grad():
        for X, y in Dataloader:
            pred = model(X)
            loss = loss_fn(pred, y)

            test_loss += loss.item() * X.size(0)
            total_samples += X.size(0)  

            correct += ((pred.argmax(1) == y).type(torch.float).sum().item())

        accuracy = correct/total_samples
        avg_loss = test_loss/total_samples

        print(f"TEST:::  correct% = {100 * accuracy} \t avg loss: {avg_loss}")


batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size = batch_size)
test_dataloader = DataLoader(test_dataset, batch_size = batch_size)

#print(next(iter(train_dataloader)))

epochs = 100
for t in range (epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, andy, loss_fn, optimizer)
    test(test_dataloader, andy, loss_fn)


#print(train_dataloader.dataset.X)


import requests
from pathlib import Path

if Path("helper_functions.py").is_file():
  print("helper_functions.py already exists, skipping download")
else:
  print("Downloading helper_functions.py")
  request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
  with open("helper_functions.py", "wb") as f:
    f.write(request.content)

from helper_functions import plot_decision_boundary


# Plot decision boundary of the model
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(andy, train_dataloader.dataset.X , train_dataloader.dataset.y  )
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(andy, test_dataloader.dataset.X , test_dataloader.dataset.y  ) 

plt.show()

