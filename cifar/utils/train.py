import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from utils.data import get_train_dataloader, get_test_dataloader
from utils.model import VGG
from utils.optim import GD
from utils.ef21 import EF21
from utils.criterions import compute_accuracy, compute_gradient_norm


def save_training_data(data, path):
    existing_data =  torch.load(path) if os.path.exists(path) else None
    if existing_data:
        for key in data:
            existing_data[key].extend(data[key])
        data = existing_data
    torch.save(data, path)


def save_model_and_optimizer(model, optimizer, ef_model, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'compression_errors': ef_model.get_compression_errors()
    }, path)

def load_model_and_optimizer(model, optimizer, ef_model, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    ef_model.load_compression_errors(checkpoint['compression_errors'])
    return model, optimizer

def find_last_checkpoint(model_dir, compression_op_name):
    checkpoints = [f for f in os.listdir(model_dir) if f.startswith(compression_op_name) and f.endswith('.pt')]
    if checkpoints:
        last_epoch = max(int(f.split('_')[-1].split('.')[0]) for f in checkpoints)
        return os.path.join(model_dir, f"{compression_op_name}_epoch_{last_epoch}.pt"), last_epoch
    return None, 0

def run_training(compression_op_name, compression_op, num_epochs, model_dir='./models', data_dir='./w', device = 'mps'):
    model = VGG().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = GD(model.parameters(), lr=0.1)
    ef_method = EF21(list(model.named_parameters()))
    ef_method.comp_operator = compression_op

    trainloader = get_train_dataloader()

    start_epoch = 0
    checkpoint_path, start_epoch = find_last_checkpoint(
        model_dir, compression_op_name)
    if checkpoint_path:
        model, optimizer = load_model_and_optimizer(
            model, optimizer, ef_method, checkpoint_path)

    losses = []
    gradient_norms = []
    accuracies = []

    for epoch in range(start_epoch, start_epoch + num_epochs):
        print(f"Starting epoch {epoch+1} with {compression_op_name}")
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, targets)
            loss.backward()

            gradient_norms.append(compute_gradient_norm(model))
            losses.append(loss.item())
            
            ef_method.step()
            optimizer.step()

        accuracies.append(compute_accuracy(model, trainloader, device))

    save_model_and_optimizer(model, optimizer, ef_method, os.path.join(
        model_dir, f"{compression_op_name}_epoch_{epoch+1}.pt"))

    data = {
        'losses': losses,
        'gradient_norms': gradient_norms,
        'accuracies': accuracies,
        'comp_factors': ef_method.comp_factors
    }
    save_training_data(data, os.path.join(
        data_dir, f"{compression_op_name}_training_data.pt"))
