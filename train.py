#!/usr/bin/env python

"""
cnn을 이용한 fashion mnist실험에서의 wandb 적용 예제입니다.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
import torch.nn.functional as F
from fashion_data import fashion

import wandb
import os


hyperparameter_defaults = dict(
    dropout = 0.5,
    channels_one = 16,
    channels_two = 32,
    batch_size = 100,
    learning_rate = 0.001,
    epochs = 2,
    )

wandb.init(config=hyperparameter_defaults, project="pytorch-cnn-example")
config = wandb.config

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        # Convolution 1
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=config.channels_one, kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=config.channels_one, out_channels=config.channels_two, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU()

        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.dropout = nn.Dropout(p=config.dropout)

        # Fully connected 1 (readout)
        self.fc1 = nn.Linear(config.channels_two*4*4, 10)

    def forward(self, x):
        # Convolution 1
        out = self.cnn1(x)
        out = self.relu1(out)

        # Max pool 1
        out = self.maxpool1(out)

        # Convolution 2
        out = self.cnn2(out)
        out = self.relu2(out)

        # Max pool 2
        out = self.maxpool2(out)

        # Resize
        # Original size: (100, 32, 7, 7)
        # out.size(0): 100
        # New out size: (100, 32*7*7)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        # Linear function (readout)
        out = self.fc1(out)

        return out

def main():
    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x/255.0 for x in [63.0, 62.1, 66.7]])

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])

    train_dataset = fashion(root='./data',
                                train=True,
                                transform=transform,
                                download=True
                               )

    test_dataset = fashion(root='./data',
                                train=False,
                                transform=transform,
                               )

    label_names = [
        "T-shirt or top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Boot"]

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=config.batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=config.batch_size,
                                              shuffle=False)


    model = CNNModel()
    wandb.watch(model)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    iter = 0
    
    train_losses, test_losses = [], []
    for epoch in range(config.epochs):
        
        running_loss = 0
        correct = 0
        
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            
            images, labels = Variable(images), Variable(labels)
                
            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()

            # Forward pass to get output/logits
            outputs = model(images)
            
            _, predicted = torch.max(outputs.data, 1)

            correct += (predicted == labels).sum()
            
            # Calculate Loss: softmax --> cross entropy loss
            loss = criterion(outputs, labels)
        
            # Getting gradients w.r.t. parameters
            loss.backward()
            
            running_loss += loss.item()

            # Updating parameters
            optimizer.step()

        train_acc = correct / len(train_loader)
        train_loss = running_loss / len(train_loader)

        running_loss = 0
        correct = 0

        model.eval()
        with torch.no_grad():
            
            for i, (images, labels) in enumerate(test_loader):
                
                images, labels = Variable(images), Variable(labels)

                outputs = model(images)
                
                _, predicted = torch.max(outputs.data, 1)

                correct += (predicted == labels).sum()
                
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                
            val_acc = correct / len(test_loader)
            val_loss = running_loss / len(test_loader)
        
        print(f"epoch {epoch} / {config.epochs - 1}\ttrain_acc : {train_acc}%, train_loss : {train_loss}, val_acc : {val_acc}, val_loss : {val_loss}")
        
        metrics = dict(
            epoch = epoch,
            train_acc = train_acc,
            train_loss = train_loss,
            val_acc = val_acc,
            val_loss = val_loss,
        )
        wandb.log(metrics)

if __name__ == '__main__':
   main()