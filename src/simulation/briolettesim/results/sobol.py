from SALib import ProblemSpec
from SALib.analyze.sobol import analyze
from SALib.sample.sobol import sample
from SALib.test_functions import Ishigami
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import stats

from results.plot_results_p2p_p2m.plot import plot_results_p2p_p2m
from utils import read_experiment_results, read_experiment_results_avg_out


# FLOATS:
# move_prob: [0, 1]
# p2p_prob: [0, 1]
# p2m_prob: [0, 1]
# random_sync_prod: [0, 0.1]
# ratio good/bad actors: [1/10, 2]


# GRAPH CHARACTERISTICS:
# Urban: m (int)
# Rural: p (float), k(int)


# INTS: (floor(float) in Rust code)
# topup_amount: [1, 55]
# tickets_given_initially: [1, 55]
# lower_bound_refill: [1, 10]
# merchant_sync_frequency: [1, 48] 


# Define the model inputs
problem = ProblemSpec({
    'num_vars': 4,
    'names': [
        'move_probability', 
        'p2p_probability', 
        'p2m_probability',
        'ratio_double_spenders_to_honest',
        # 'random_sync_probability',
        # 'top_up_amount',
        # 'tickets_given_right_away',
        # 'tickets_lower_bound_to_sync',
        # 'merchant_sync_frequency'
    ],
    'bounds': [
        # CONTINUOUS
        [0, 1], # move_prob
        [0.1, 0.5], # p2p_prob
        [0.1, 0.5], # p2m_prob

        [0.1, 4], # bad_actors / good_actors ratio
        # [0.01, 0.1], # random_sync_prob

        # DISCRETE
        # [1, 55], # top_up_amount
        # [1, 55], # tickets_given_initially / ticket_refill
        # [1, 10], # tickets_lower_bound_to_sync
        # [1, 48], # merchant_sync_frequency
        # 
        # 
        # [0, 1], # move_prob 

        # [1, 55], # top_up_amount
        # [1, 55], # tickets_given_initially / ticket_refill
        # [1, 10], # tickets_lower_bound_to_sync

        # [1, 48], # merchant_sync_frequency
    ],
    "outputs": [
        "ratio_of_double_spent_coins", 
        "ratio_of_double_spenders_caught", 
        "global_to_local_epoch_diffs_mean", 
        "global_to_local_epoch_diffs_std", 
        "global_to_local_epoch_diffs_max"
    ],
    ## take mean + std of the outputs -> if the std is small compared to the mean  = good, no need to increase the number of experiments
    ## otherwise, increase the number of experiments
})



def generate():
    # Generate samples
    # n = 128
    # d = 2
    # n * (2 * d + 2)
    param_values = sample(problem, 128)

    with open("sobol_params_4d.txt", 'w') as file:
        for experiment_param_set in param_values:
            for param in experiment_param_set:
                file.write(str(param) + ' ')
            file.write('\n')
    
    print(param_values.shape)

# TODO: fix paths to the results
def evaluate():
    folder_path = "/home/fanlgrp/Projects/briolette_cbdc_paper/src/simulation/briolettesim/results/sobol_results/rural"
    
    # Read parameters from sobol_params.txt
    params = []
    
    with open("/home/fanlgrp/Projects/briolette_cbdc_paper/src/simulation/briolettesim/results/sobol_params.txt", 'r') as f:
        for line in f:
            params.append([float(x) for x in line.strip().split()])
    
    params = np.array(params)
    
    # Dictionary to store arrays by experiment ID
    experiments = {}
    print("Number of files to be processed:", len(os.listdir(folder_path)))
    for filename in os.listdir(folder_path):
        if not os.path.isfile(os.path.join(folder_path, filename)):
            continue
            
        # Extract experiment ID from filename
        exp_id = int(filename.split('expid_')[1].split('.')[0])
        
        # Read arrays from file
        double_spent_arrays, spenders_caught_arrays, global_to_local_epoch_diffs_mean_arrays, global_to_local_epoch_std_arrays, global_to_local_epoch_max_arrays = read_experiment_results_avg_out(os.path.join(folder_path, filename))

        if exp_id not in experiments:
            experiments[exp_id] = {
                'double_spent': double_spent_arrays,
                'spenders_caught': spenders_caught_arrays,
                'diff_epoch_mean': global_to_local_epoch_diffs_mean_arrays,
                'diff_epoch_std': global_to_local_epoch_std_arrays,
                'diff_epoch_max': global_to_local_epoch_max_arrays,
                'params': params[exp_id] if exp_id < len(params) else None
            }
        else:
            experiments[exp_id]['double_spent'] = np.vstack([
                experiments[exp_id]['double_spent'], 
                double_spent_arrays
            ])
            experiments[exp_id]['spenders_caught'] = np.vstack([
                experiments[exp_id]['spenders_caught'], 
                spenders_caught_arrays
            ])

    # Calculate averages and standard deviations
    results = []
    for exp_id, data in experiments.items():
        if data['params'] is None:
            continue
            
        # For each time step, calculate mean and std
        double_spent_means = []
        double_spent_stds = []
        spenders_caught_means = []
        spenders_caught_stds = []

        epoch_diff_mean_of_means = []
        epoch_diff_mean_of_stds = []
        epoch_diff_mean_of_max = []
        
        # Get maximum length of arrays
        max_len_ds = max(len(arr) for arr in data['double_spent'])
        max_len_sc = max(len(arr) for arr in data['spenders_caught'])
        max_len_epoch_mean = max(len(arr) for arr in data['diff_epoch_mean'])
        
        # Calculate statistics for each time step
        for i in range(max_len_ds):
            values = [arr[i] for arr in data['double_spent'] if i < len(arr)]
            if values:
                double_spent_means.append(np.mean(values))
                double_spent_stds.append(np.std(values))
                
        for i in range(max_len_sc):
            values = [arr[i] for arr in data['spenders_caught'] if i < len(arr)]
            if values:
                spenders_caught_means.append(np.mean(values))
                spenders_caught_stds.append(np.std(values))

        for i in range(max_len_epoch_mean):
            values = [arr[i] for arr in data['diff_epoch_mean'] if i < len(arr)]
            if values:
                epoch_diff_mean_of_means.append(np.mean(values))
            

            values = [arr[i] for arr in data['diff_epoch_std'] if i < len(arr)]
            if values:            
                epoch_diff_mean_of_stds.append(np.mean(values))

            

            values = [arr[i] for arr in data['diff_epoch_max'] if i < len(arr)]
            if values:
                epoch_diff_mean_of_max.append(np.mean(values))
        

        
        results.append({
            'exp_id': exp_id,
            'params': data['params'],
            'double_spent_means': double_spent_means,
            'double_spent_stds': double_spent_stds,
            'spenders_caught_means': spenders_caught_means,
            'spenders_caught_stds': spenders_caught_stds,
            'diff_epoch_mean_mean': epoch_diff_mean_of_means,
            'diff_epoch_std_mean': epoch_diff_mean_of_stds,
            'diff_epoch_max_mean': epoch_diff_mean_of_max,
        })

    # # Create figure with 6 subplots (2x3)
    # plt.figure(figsize=(20, 15))

    # # Create figure with 2 subplots (2x1)
    # plt.figure(figsize=(10, 20))
    
    # Plot 1: Double spent ratios over time
    # plt.subplot(2, 1, 1)
    # for idx, result in enumerate(results):
    #     # if (idx % 100 == 0):
    #         x = range(len(result['double_spent_means']))
    #         means = result['double_spent_means']
    #         stds = result['double_spent_stds']
            
    #         plt.plot(x, means, label=f'Exp {round(float(result["params"][0]), 2), round(float(result["params"][1]), 2)}')
    #         plt.fill_between(x, 
    #                         np.array(means) - np.array(stds),
    #                         np.array(means) + np.array(stds),
    #                         alpha=0.2)
    
    # plt.title('Double Spent Coins Ratio Over Time')
    # plt.xlabel('Time Step')
    # plt.ylabel('Ratio')
    # # plt.legend()
    
    # # Plot 2: Spenders caught ratios over time
    # plt.subplot(2, 1, 2)
    # for idx, result in enumerate(results):
    #     # if (idx % 100 == 0):
    #         x = range(len(result['spenders_caught_means']))
    #         means = result['spenders_caught_means']
    #         stds = result['spenders_caught_stds']
            
    #         plt.plot(x, means, label=f'Exp {round(float(result["params"][0]), 2), round(float(result["params"][1]), 2)}')
    #         plt.fill_between(x, 
    #                         np.array(means) - np.array(stds),
    #                         np.array(means) + np.array(stds),
    #                         alpha=0.2)
    
    # plt.title('Double Spenders Caught Ratio Over Time')
    # plt.xlabel('Time Step')
    # plt.ylabel('Ratio')
    # plt.legend()


    # plot_results_p2p_p2m(results)
    

if __name__ == "__main__":
    generate()  
    # evaluate()

# set malicious / good users ratio to a way smaller number than 2:1 (0.1)?
# 8 + 8 plots (urban + rural)
# fit the linear regression, get R^2
# put graphs in a paper, try to explain them