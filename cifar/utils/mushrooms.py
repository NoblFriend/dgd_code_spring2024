from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class MushroomsDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float64)
        self.y = torch.tensor(y, dtype=torch.float64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def _load_mushrooms_dataset(dataset_path="./data/mushrooms.txt", test_size=0.2):
    # Загрузка и преобразование данных
    data = load_svmlight_file(dataset_path)
    X, y = data[0].toarray(), data[1]
    y = y - 1  # Преобразование меток
    
    # Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Создание объектов Dataset
    train_dataset = MushroomsDataset(X_train, y_train)
    test_dataset = MushroomsDataset(X_test, y_test)

    return train_dataset, test_dataset

def get_train_data(dataset_path="./data/mushrooms.txt", test_size=0.2):
    train_dataset, _ = _load_mushrooms_dataset(dataset_path, test_size)
    return train_dataset


def get_test_dataloader(dataset_path="./data/mushrooms.txt", batch_size=10000000, test_size=0.2):
    _, test_dataset = _load_mushrooms_dataset(dataset_path, test_size)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return test_dataloader

def get_train_dataloader(dataset_path="./data/mushrooms.txt", batch_size=10000000, test_size=0.2):
    test_dataset, _ = _load_mushrooms_dataset(dataset_path, test_size)
    train_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_dataloader