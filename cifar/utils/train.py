import torch
import os

def save_training_data(data, path):
    existing_data = load_training_data(path) if os.path.exists(path) else None
    if existing_data:
        for key in data:
            existing_data[key].extend(data[key])
        data = existing_data
    torch.save(data, path)

def load_training_data(path):
    return torch.load(path)

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