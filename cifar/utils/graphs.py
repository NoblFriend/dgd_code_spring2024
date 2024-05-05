import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd

def plot_training_results(compression_operators, data_dir="./w"):
    plt.figure(figsize=(18, 6))
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

def make_plot(ax, data, plot_type, label, plot_func, window=100):
    x_values = np.cumsum(data['comp_factors'])[::len(data['comp_factors'])//len(data[plot_type])]
    window = max(1, int((window * len(x_values))/(x_values[-1])))

    # Серия данных и применение скользящего среднего и стандартной девиации с минимальным периодом 1
    data_series = pd.Series(data[plot_type])
    rolling_mean = data_series.rolling(window=window, min_periods=1).mean()
    rolling_std = data_series.rolling(window=window, min_periods=1).std()

    # Обрезка x_values, если размер последнего блока меньше window
    x_values = x_values[:len(rolling_mean)]

    # Построение графика в зависимости от типа функции
    if plot_func == 'semilogy':
        ax.semilogy(x_values, rolling_mean, label=f'{label} {plot_type}')
    else:
        ax.plot(x_values, rolling_mean, label=f'{label} {plot_type}')

    # Заполнение области вокруг среднего значением стандартной девиации, уменьшенной вдвое
    ax.fill_between(x_values, rolling_mean-rolling_std/2, rolling_mean+rolling_std/2, alpha=0.5)

    # Настройка осей и легенды
    ax.set_xlabel('Cumulative Comp Factor')
    ax.set_ylabel(plot_type.capitalize())
    ax.legend()
    ax.grid(True)