import matplotlib.pyplot as plt
from utils.train import load_training_data

def plot_training_results(compression_operators, data_dir = "./w"):
    plt.figure(figsize=(18, 6))

    # Plot gradient norms
    plt.subplot(1, 3, 1)
    for op_name in compression_operators.keys():
        data = load_training_data(f"{data_dir}/{op_name}_training_data.pt")
        plt.semilogy(data['gradient_norms'], label=f'{op_name} Gradient Norms')
    plt.xlabel('Step Number')
    plt.ylabel('Gradient Norm')
    plt.title('Gradient Norms over Training Steps')
    plt.legend()
    plt.grid(True)

    # Plot losses
    plt.subplot(1, 3, 2)
    for op_name in compression_operators.keys():
        data = load_training_data(f"{data_dir}/{op_name}_training_data.pt")
        plt.semilogy(data['losses'], label=f'{op_name} Losses')
    plt.xlabel('Step Number')
    plt.ylabel('Loss')
    plt.title('Losses over Training Steps')
    plt.legend()
    plt.grid(True)

    # Plot weights norms
    plt.subplot(1, 3, 3)
    for op_name in compression_operators.keys():
        data = load_training_data(f"{data_dir}/{op_name}_training_data.pt")
        plt.plot(data['accuracies'], label=f'{op_name} Accuracy')
    plt.xlabel('Step Number')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Training Steps')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()