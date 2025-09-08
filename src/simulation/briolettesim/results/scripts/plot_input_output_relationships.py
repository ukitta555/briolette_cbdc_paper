import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os


# TODO: 
# View how many time steps it takes to reach 99% of the double spenders caught  and plot it 
# have a look at the covariance matrix of the inputs and outputs


def plot_input_output_relationships(results, time_step=-1, output_dir='plots'):
    """
    Plot relationships between inputs and outputs at a specific time step.
    
    Args:
        results: List of dictionaries containing experiment results
        time_step: Index of the time step to plot (-1 for last, -2 for second to last, etc.)
        output_dir: Directory where plots will be saved
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with 12 subplots (4x3)
    plt.figure(figsize=(24, 20))
    
    input_names = ['Move Probability', 'P2P Probability', 
                   'P2M Probability', 'Ratio Double Spenders to Honest']
    
    print("len of double spent means", len(results[0]['double_spent_means']))
    # Plot relationships for double spent ratio
    for i, input_name in enumerate(input_names):
        plt.subplot(4, 3, 3*i + 1)
        x = [result['params'][i] for result in results]
        y = [result['double_spent_means'][time_step] for result in results]
        
        # Calculate correlation coefficient and p-value
        correlation, p_value = stats.pearsonr(x, y)
        
        # Calculate R-squared
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        y_pred = p(x)
        r_squared = 1 - (np.sum((np.array(y) - y_pred) ** 2) / 
                        np.sum((np.array(y) - np.mean(y)) ** 2))
        
        plt.scatter(x, y, alpha=0.5)
        
        # Add trend line
        plt.plot(x, p(x), "r--", alpha=0.8)
        
        plt.xlabel(input_name)
        plt.ylabel('Double Spent Ratio')
        
        # Plot relationships for spenders caught ratio
        plt.subplot(4, 3, 3*i + 2)
        # y = [result['spenders_caught_means'][time_step] / (result['params'][0] * result['params'][1] * result['params'][2]) for result in results]
        
        
        y = [result['spenders_caught_means'][time_step]  for result in results]

        ratios = [result['params'][3] for result in results] 
        actual_ratio_of_spenders_caught = [y[idx] / (el / (1 + el)) for idx, el in enumerate(ratios)]



        # Calculate correlation coefficient and p-value
        correlation, p_value = stats.pearsonr(x, actual_ratio_of_spenders_caught)
        
        # Calculate R-squared
        z = np.polyfit(x, actual_ratio_of_spenders_caught, 1)
        p = np.poly1d(z)
        y_pred = p(x)
        r_squared = 1 - (np.sum((np.array(actual_ratio_of_spenders_caught) - y_pred) ** 2) / 
                        np.sum((np.array(actual_ratio_of_spenders_caught) - np.mean(actual_ratio_of_spenders_caught)) ** 2))
        
        # indices_with_less_than_300 = [idx for idx, el in enumerate(actual_ratio_of_spenders_caught) if el < 300]
        
        # x_new = []
        # actual_ratio_of_spenders_caught_new = []
        # for idx in indices_with_less_than_300:
        #     x_new.append(x[idx])
        #     actual_ratio_of_spenders_caught_new.append(actual_ratio_of_spenders_caught[idx])

        # plt.scatter(x_new, actual_ratio_of_spenders_caught_new, alpha=0.5)

        plt.scatter(x, actual_ratio_of_spenders_caught, alpha=0.5)
        
        # Add trend line
        plt.plot(x, p(x), "r--", alpha=0.8)
        
        plt.xlabel(input_name)
        plt.ylabel('Spenders Caught Ratio')

        # Plot relationships for global-local mean
        plt.subplot(4, 3, 3*i + 3)
        x = [result['params'][i] for result in results]
        y = [result['diff_epoch_mean_mean'][time_step] for result in results]
        
        # Calculate correlation coefficient and p-value
        correlation, p_value = stats.pearsonr(x, y)
        
        # Calculate R-squared
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        y_pred = p(x)
        r_squared = 1 - (np.sum((np.array(y) - y_pred) ** 2) / 
                        np.sum((np.array(y) - np.mean(y)) ** 2))
        
        plt.scatter(x, y, alpha=0.5)
        
        # Add trend line
        plt.plot(x, p(x), "r--", alpha=0.8)
        
        plt.xlabel(input_name)
        plt.ylabel('Global-Local Mean')
    
    plt.tight_layout()
    # Calculate the actual time step number (positive number)
    actual_time_step = len(results[0]['double_spent_means']) + time_step if time_step < 0 else time_step

    plt.savefig(os.path.join(output_dir, f'{actual_time_step}_input_output_relationships.png'))
    plt.close() 