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

def compute_gradient_norm(model):
    """ Compute the overall norm of the gradients for the model. """
    total_grad_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total_grad_norm += (param.grad.data.norm(2) ** 2).item()  # Суммируем квадраты норм
    total_grad_norm = total_grad_norm ** 0.5  # Извлекаем квадратный корень из суммы квадратов
    return total_grad_norm
