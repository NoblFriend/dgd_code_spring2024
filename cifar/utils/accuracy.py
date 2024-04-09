import torch
import numpy as np

def compute_accuracy(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to('mps'), target.to('mps')
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    model.train()
    return correct / total

def compute_accuracy_mushrooms(model, data):
    model.eval()
    with torch.no_grad():  # Отключаем вычисление градиентов
            outputs = model(data.X)
            predictions = torch.round(torch.sigmoid(outputs)).squeeze()
            correct = (predictions == data.y).sum().item()
            total = data.y.size(0)
    model.train()  
    return correct / total if total > 0 else 0

