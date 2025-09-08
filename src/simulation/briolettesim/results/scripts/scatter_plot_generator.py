from SALib import ProblemSpec
from SALib.sample.sobol import sample
from SALib.test_functions import Ishigami
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import stats
import argparse
from datetime import datetime

from plot_input_output_relationships import plot_input_output_relationships
from utils import read_experiment_results, read_experiment_results_avg_out
from sobol_analysis import perform_sobol_analysis

# python sobol.py --mode generate --sample-size 128 --output-params-file sobol_params.txt
# python3 scripts/scatter_plot_generator.py --mode evaluate --results-dir /path/to/results --output-dir plots --num-time-steps 30


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
        [0.05, 0.7], # move_prob
        [0.1, 0.5], # p2p_prob
        [0.1, 0.5], # p2m_prob
        [0.1, 7], # bad_actors / good_actors ratio
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



def generate(args):
    # Generate samples
    param_values = sample(problem, args.sample_size)

    with open(args.output_params_file, 'w') as file:
        for experiment_param_set in param_values:
            for param in experiment_param_set:
                file.write(str(param) + ' ')
            file.write('\n')
    
    print(f"Generated {param_values.shape[0]} parameter sets")

def evaluate(args):
    # Create timestamped output directory one level above
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                            'plots', f'plots_sobol_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Read parameters from sobol_params.txt
    params = []
    
    with open(args.input_params_file, 'r') as f:
        for line in f:
            params.append([float(x) for x in line.strip().split()])
    
    params = np.array(params)

    
    # Dictionary to store arrays by experiment ID
    experiments = {}
    print(f"Number of files to be processed in {args.results_dir}:", len(os.listdir(args.results_dir)))
    for filename in os.listdir(args.results_dir):
        if not os.path.isfile(os.path.join(args.results_dir, filename)):
            continue
            
        if 'flamegraph' in filename:
            continue
        # Extract experiment ID from filename
        try:
            exp_id = int(filename.split('expid_')[1].split('.')[0])
        except:
            exp_id = int(filename.split('expid_')[1].split('_')[0])

        # Read arrays from file
        double_spent_arrays, spenders_caught_arrays, global_to_local_epoch_diffs_mean_arrays, global_to_local_epoch_std_arrays, global_to_local_epoch_max_arrays = read_experiment_results_avg_out(os.path.join(args.results_dir, filename))

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
                spenders_caught_stds.append(np.mean(values))

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
    
    # Generate plots for multiple time steps
    # num_time_steps = min(args.num_time_steps, len(results[0]['double_spent_means']))
    # for time_step in range(-1, -num_time_steps-1, -1):
    #     plot_input_output_relationships(results, time_step, output_dir)
    
    plot_input_output_relationships(results, time_step=-1, output_dir=output_dir)
    
    print("Performing Sobol analysis...")
    if args.perform_sobol:
        perform_sobol_analysis(results, output_dir=output_dir, results_dir=args.results_dir)
    print("Sobol analysis complete. Results saved in:", output_dir)


def main():
    parser = argparse.ArgumentParser(description='Sobol Sensitivity Analysis for CBDC Simulation')
    
    # Common arguments
    parser.add_argument('--mode', choices=['generate', 'evaluate'], required=True,
                      help='Operation mode: generate parameters or evaluate results')
    
    # Generate mode arguments
    parser.add_argument('--sample-size', type=int, default=128,
                      help='Number of parameter sets to generate (default: 128)')
    parser.add_argument('--output-params-file', type=str, default='sobol_params.txt',
                      help='Output file for generated parameters (default: sobol_params.txt)')
    
    # Evaluate mode arguments
    parser.add_argument('--input-params-file', type=str, default='sobol_params.txt',
                      help='Input file containing parameter sets (default: sobol_params.txt)')
    parser.add_argument('--results-dir', type=str, default='123321',
                      help='Directory containing simulation results')
    parser.add_argument('--num-time-steps', type=int, default=30,
                      help='Number of time steps to analyze (default: 30)')
    parser.add_argument('--perform-sobol', action='store_true',
                      help='Perform Sobol sensitivity analysis')
    
    args = parser.parse_args()


    
    if args.mode == 'generate':
        generate(args)
    else:  # evaluate mode
        evaluate(args)

if __name__ == "__main__":
    main()

# set malicious / good users ratio to a way smaller number than 2:1 (0.1)?
# 8 + 8 plots (urban + rural)
# fit the linear regression, get R^2
# put graphs in a paper, try to explain them