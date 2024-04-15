import os
from itertools import zip_longest

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from utils.data import get_train_dataloader, get_test_dataloader
from utils.model import VGG
from utils.optim import GD
from utils.worker import setup_workers
from utils.accuracy import compute_accuracy

# Папка для сохранения графиков
graphs_dir = "./graphs"
os.makedirs(graphs_dir, exist_ok=True)


if __name__ == '__main__':
    model = VGG().to('mps')
    criterion = nn.CrossEntropyLoss()
    optimizer = GD(model.parameters(), lr=0.01)

    testloader = get_test_dataloader()
    trainloader = get_train_dataloader()

    num_epochs = 10
    accuracies = []
    max_idx = 0
    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch+1}") 
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            print(f"step {i}")

            model.zero_grad()

            output = model(inputs)
            loss = criterion(output, labels)
            loss.backward()

            optimizer.step()


        accuracies.append( compute_accuracy(model, testloader))
        
plt.figure(figsize=(7, 4))
plt.plot(accuracies)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy vs. Epoch")
plt.grid(True)
plt.show()

