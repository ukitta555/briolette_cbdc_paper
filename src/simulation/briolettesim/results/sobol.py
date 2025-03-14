from SALib import ProblemSpec
from SALib.analyze.sobol import analyze
from SALib.sample.sobol import sample
from SALib.test_functions import Ishigami
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import stats

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
    'num_vars': 2,
    'names': [
        # 'move_probability', 
        # 'p2p_probability', 
        # 'p2m_probability',
        'ratio_double_spenders_to_honest',
        'random_sync_probability',
        # 'top_up_amount',
        # 'tickets_given_right_away',
        # 'tickets_lower_bound_to_sync',
        # 'merchant_sync_frequency'
    ],
    'bounds': [
        # CONTINUOUS
        # [0, 1], # move_prob
        # [0, 1], # p2p_prob
        # [0, 1], # p2p_prob

        # prioritize these params!
        ####
        [0.1, 0.5], # bad_actors / good_actors ratio
        [0.01, 0.1], # random_sync_prob
        ####
        # DISCRETE
        # [1, 55], # top_up_amount
        # [1, 55], # tickets_given_initially / ticket_refill
        # [1, 10], # tickets_lower_bound_to_sync
        # [1, 48], # merchant_sync_frequency 
    ],
    "outputs": ["ratio_of_double_spent_coins", "ratio_of_double_spenders_caught"],
    ## take mean + std of the outputs -> if the std is small compared to the mean  = good, no need to increase the number of experiments
    ## otherwise, increase the number of experiments
})



def generate():
    # Generate samples
    # n = 128
    # d = 2
    # n * (2 * d + 2)
    param_values = sample(problem, 128)

    with open("sobol_params.txt", 'w') as file:
        for experiment_param_set in param_values:
            for param in experiment_param_set:
                file.write(str(param) + ' ')
            file.write('\n')
    
    print(param_values.shape)


def evaluate():
    folder_path = "/home/vladyslav/VSCodeProjects/briolette/src/simulation/briolettesim/results/sobol_results/rural"
    
    # Read parameters from sobol_params.txt
    params = []
    with open("/home/vladyslav/VSCodeProjects/briolette/src/simulation/briolettesim/results/sobol_params.txt", 'r') as f:
        for line in f:
            params.append([float(x) for x in line.strip().split()])
    params = np.array(params)
    
    # Dictionary to store arrays by experiment ID
    experiments = {}
    
    for filename in os.listdir(folder_path):
        if not os.path.isfile(os.path.join(folder_path, filename)):
            continue
            
        # Extract experiment ID from filename
        exp_id = int(filename.split('expid_')[1].split('.')[0])
        
        # Read arrays from file
        double_spent_arrays, spenders_caught_arrays = read_experiment_results_avg_out(
            os.path.join(folder_path, filename)
        )
        
        if exp_id not in experiments:
            experiments[exp_id] = {
                'double_spent': double_spent_arrays,
                'spenders_caught': spenders_caught_arrays,
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
        
        # Get maximum length of arrays
        max_len_ds = max(len(arr) for arr in data['double_spent'])
        max_len_sc = max(len(arr) for arr in data['spenders_caught'])
        
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
        
        results.append({
            'exp_id': exp_id,
            'params': data['params'],
            'double_spent_means': double_spent_means,
            'double_spent_stds': double_spent_stds,
            'spenders_caught_means': spenders_caught_means,
            'spenders_caught_stds': spenders_caught_stds
        })

    # # Create figure with 6 subplots (2x3)
    # plt.figure(figsize=(20, 15))

    # Create figure with 2 subplots (2x1)
    plt.figure(figsize=(10, 20))
    
    # Plot 1: Double spent ratios over time
    plt.subplot(2, 1, 1)
    for idx, result in enumerate(results):
        # if (idx % 100 == 0):
            x = range(len(result['double_spent_means']))
            means = result['double_spent_means']
            stds = result['double_spent_stds']
            
            plt.plot(x, means, label=f'Exp {round(float(result["params"][0]), 2), round(float(result["params"][1]), 2)}')
            plt.fill_between(x, 
                            np.array(means) - np.array(stds),
                            np.array(means) + np.array(stds),
                            alpha=0.2)
    
    plt.title('Double Spent Coins Ratio Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Ratio')
    # plt.legend()
    
    # Plot 2: Spenders caught ratios over time
    plt.subplot(2, 1, 2)
    for idx, result in enumerate(results):
        # if (idx % 100 == 0):
            x = range(len(result['spenders_caught_means']))
            means = result['spenders_caught_means']
            stds = result['spenders_caught_stds']
            
            plt.plot(x, means, label=f'Exp {round(float(result["params"][0]), 2), round(float(result["params"][1]), 2)}')
            plt.fill_between(x, 
                            np.array(means) - np.array(stds),
                            np.array(means) + np.array(stds),
                            alpha=0.2)
    
    plt.title('Double Spenders Caught Ratio Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Ratio')
    # plt.legend()
    
    # # Plot 3: Double spent ratio vs ratio_double_spenders_to_honest
    # plt.subplot(2, 3, 3)
    # x = [result['params'][0] for result in results]
    # y = [result['double_spent_means'][-1] for result in results]
    # plt.scatter(x, y)
    # plt.title('Final Double Spent Ratio vs DS/Honest Ratio')
    # plt.xlabel('Double Spenders to Honest Ratio')
    # plt.ylabel('Final Double Spent Ratio')
    
    # # Plot 4: Double spent ratio vs random_sync_probability
    # plt.subplot(2, 3, 4)
    # x = [result['params'][1] for result in results]
    # y = [result['double_spent_means'][-1] for result in results]
    # plt.scatter(x, y)
    # plt.title('Final Double Spent Ratio vs Sync Probability')
    # plt.xlabel('Random Sync Probability')
    # plt.ylabel('Final Double Spent Ratio')
    
    # # Plot 5: Spenders caught vs ratio_double_spenders_to_honest
    # plt.subplot(2, 3, 5)
    # x = [result['params'][0] for result in results]
    # y = [result['spenders_caught_means'][-1] for result in results]
    # plt.scatter(x, y)
    # plt.title('Final Spenders Caught vs DS/Honest Ratio')
    # plt.xlabel('Double Spenders to Honest Ratio')
    # plt.ylabel('Final Spenders Caught Ratio')
    
    # # Plot 6: Spenders caught vs random_sync_probability
    # plt.subplot(2, 3, 6)
    # x = [result['params'][1] for result in results]
    # y = [result['spenders_caught_means'][-1] for result in results]
    # plt.scatter(x, y)
    # plt.title('Final Spenders Caught vs Sync Probability')
    # plt.xlabel('Random Sync Probability')
    # plt.ylabel('Final Spenders Caught Ratio')
    
    plt.tight_layout()
    plt.savefig('experiment_results_all.png')
    plt.close()

if __name__ == "__main__":
    # generate()
    evaluate()