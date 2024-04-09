import os
from itertools import zip_longest

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from utils.data import get_train_data, get_test_dataloader
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

    workers = setup_workers(
        model=model,
        criterion=criterion,
        dataset=get_train_data(), 
        num_workers=1,
        batch_size=128
    )

    num_epochs = 10
    accuracies = []
    max_idx = 0
    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch+1}")
        for idx, grads_tuple in enumerate(zip_longest(*[worker.gradient_generator() for worker in workers])):
            print(f"step {idx}")
            max_idx = max(idx, max_idx)
            model.zero_grad()
            num_grads = 0
            for grads in filter(None, grads_tuple):
                num_grads += 1
                alpha = 1/(num_grads)
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        if  param.grad == None:
                            param.grad = grads[name]
                        else:
                            param.grad = alpha * grads[name] + (1-alpha)*param.grad

            optimizer.step()


        accuracies.append( compute_accuracy(model, testloader))
        # Сохранение графика accuracy
        plt.figure(figsize=(10, 6))
        plt.plot(accuracies, label="Accuracy", marker='o')
        plt.xlabel("Step")
        plt.ylabel("Accuracy")
        plt.title("Accuracy vs. Steps")
        plt.legend()
        plt.grid(True)
        
        # Сохранение графика в файл
        plt.savefig(os.path.join(graphs_dir, f"epoch-{epoch+1}.png"))
        plt.close()
        print(f"Epoch {epoch+1}/{num_epochs} completed.")

