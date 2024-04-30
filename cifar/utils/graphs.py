import matplotlib.pyplot as plt
import numpy as np
import torch


def make_plot(ax, data, plot_type, label, plot_func):
    x_values = np.cumsum(data['comp_factors'])[::len(data['comp_factors'])//len(data[plot_type])]
    if plot_func == 'semilogy':
        ax.semilogy(x_values, data[plot_type], label=f'{label} {plot_type}')
    else:
        ax.plot(x_values, data[plot_type], label=f'{label} {plot_type}')
    ax.set_xlabel('Cumulative Comp Factor')
    ax.set_ylabel(plot_type.capitalize())
    ax.legend()
    ax.grid(True)

def plot_training_results(compression_operators, data_dir="./w"):
    plt.figure(figsize=(18, 6))
    # plot_types = ['gradient_norms', 'losses', 'accuracies']
    plot_types = ['gradient_norms', 'losses', 'accuracies']

    for i, plot_type in enumerate(plot_types, 1):
        ax = plt.subplot(1, 3, i)
        for op_name in compression_operators.keys():
            data = torch.load(f"{data_dir}/{op_name}_training_data.pt")
            plot_func = 'semilogy' if plot_type != 'accuracies' else 'plot'
            make_plot(ax, data, plot_type, op_name, plot_func)
        ax.set_title(f'{plot_type.capitalize()} over Training Steps')

    plt.tight_layout()
    plt.show()



# def plot_training_results(compression_operators, data_dir = "./w"):
#     plt.figure(figsize=(18, 6))

#     # Plot gradient norms
#     plt.subplot(1, 3, 1)
#     for op_name in compression_operators.keys():
#         data = load_training_data(f"{data_dir}/{op_name}_training_data.pt")
#         plt.semilogy(data['gradient_norms'], label=f'{op_name} Gradient Norms')
#     plt.xlabel('Step Number')
#     plt.ylabel('Gradient Norm')
#     plt.title('Gradient Norms over Training Steps')
#     plt.legend()
#     plt.grid(True)

#     # Plot losses
#     plt.subplot(1, 3, 2)
#     for op_name in compression_operators.keys():
#         data = load_training_data(f"{data_dir}/{op_name}_training_data.pt")
#         plt.semilogy(data['losses'], label=f'{op_name} Losses')
#     plt.xlabel('Step Number')
#     plt.ylabel('Loss')
#     plt.title('Losses over Training Steps')
#     plt.legend()
#     plt.grid(True)

#     # Plot weights norms
#     plt.subplot(1, 3, 3)
#     for op_name in compression_operators.keys():
#         data = load_training_data(f"{data_dir}/{op_name}_training_data.pt")
#         plt.plot(data['accuracies'], label=f'{op_name} Accuracy')
#     plt.xlabel('Step Number')
#     plt.ylabel('Accuracy')
#     plt.title('Accuracy over Training Steps')
#     plt.legend()
#     plt.grid(True)

#     plt.tight_layout()
#     plt.show()