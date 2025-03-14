import matplotlib.pyplot as plt
from tabulate import tabulate
import numpy as np

from results.utils import read_experiment_results


# Plotting function
def plot_experiment_results(file_path):
    array_x, array_y, array_z, float_array, array_3, array_4 = read_experiment_results(file_path)

    # Plot each position in the triplets
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(array_x)), array_x, marker='o', label='Mean')
    plt.plot(range(len(array_y)), array_y, marker='o', label='Std')
    # plt.plot(range(len(array_z)), array_z, marker='o', label='Max')



    plt.title("Global - Local stats")
    plt.xlabel("Simulation step (1 step = 1 hour)")
    plt.ylabel("Measurement value")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot the float array
    plt.figure(figsize=(10, 5))
    x_values = list(range(len(float_array)))
    plt.plot(x_values, float_array, marker='o', linestyle='-', color='r', label='Intra Std of Local Epochs')
    plt.title("Intra Std of Local Epochs")
    plt.xlabel("Simulation step (1 step = 1 hour)")
    plt.ylabel("Measurement value")
    plt.legend()
    plt.grid(True)



    plt.show()


    # Plot the float array
    plt.figure(figsize=(10, 5))
    x_values = list(range(len(array_3)))
    plt.plot(x_values, array_3, marker='o', linestyle='-', color='r', label='123')
    plt.title("")
    plt.xlabel("Simulation step (1 step = 1 hour)")
    plt.ylabel("Measurement value")
    plt.legend()
    plt.grid(True)

    plt.show()


    # Plot the float array
    plt.figure(figsize=(10, 5))
    x_values = list(range(len(array_4)))
    plt.plot(x_values, array_4, marker='o', linestyle='-', color='r', label='123')
    plt.title("")
    plt.xlabel("Simulation step (1 step = 1 hour)")
    plt.ylabel("Measurement value")
    plt.legend()
    plt.grid(True)

    plt.show()

    
plot_experiment_results('experiment_results_model_Rural_ratio_2_topup_15_tickets_8_config_64.txt')
