import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from datetime import datetime

from utils import read_experiment_results_avg_out

def plot_counterfeit_vs_malicious(results, output_dir):
    """
    Plot relationship between counterfeit ratio and malicious to honest ratio
    """
    plt.figure(figsize=(10, 7))
    
    # Collect data points for regression
    ratios = []
    double_spents = []
    
    for result in results:
        # Get the last value for double spent ratio
        double_spent = result['double_spent_means'][-1]
        # Get the ratio parameter (index 3 in params)
        ratio = result['params'][3]
        
        ratios.append(ratio)
        double_spents.append(double_spent)
        
    plt.scatter(ratios, double_spents, alpha=0.2, s=50, color='tab:blue')  # Using specified hex color
    
    # Calculate regression line
    z = np.polyfit(ratios, double_spents, 1)
    p = np.poly1d(z)
    
    # Plot regression line
    x_range = np.linspace(min(ratios), max(ratios), 100)
    plt.plot(x_range, p(x_range), "r--", alpha=0.8, linewidth=5.0)
    
    plt.xlabel("Malicious to Honest Ratio", fontsize=26)
    plt.ylabel("Counterfeit", fontsize=26)
    plt.grid(True)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    
    # Increase font size for all text elements
    plt.rcParams.update({'font.size': 22})
    
    plt.tight_layout()
    
    # Save both PDF and PNG versions
    plt.savefig(os.path.join(output_dir, 'counterfeit_vs_malicious.pdf'), 
                bbox_inches='tight', dpi=1000)
    plt.savefig(os.path.join(output_dir, 'counterfeit_vs_malicious.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()

def plot_global_local_vs_p2p(results, output_dir):
    """
    Plot relationship between global-local mean difference and p2p probability
    """
    plt.figure(figsize=(10, 7))
    
    # Collect data points for regression
    p2p_probs = []
    global_local_means = []
    
    for result in results:
        # Get the last value for global-local mean
        global_local_mean = result['diff_epoch_mean_mean'][-1]
        # Get the p2p probability parameter (index 1 in params)
        p2p_prob = result['params'][1]
        
        p2p_probs.append(p2p_prob)
        global_local_means.append(global_local_mean)
        
    plt.scatter(p2p_probs, global_local_means, alpha=0.2, s=50, color='tab:blue')  # Using specified hex color
    
    # Calculate regression line
    z = np.polyfit(p2p_probs, global_local_means, 1)
    p = np.poly1d(z)
    
    # Plot regression line
    x_range = np.linspace(min(p2p_probs), max(p2p_probs), 100)
    plt.plot(x_range, p(x_range), "r--", alpha=0.8, linewidth=5.0)
    
    plt.xlabel("P2P Probability", fontsize=26)
    plt.ylabel("Global-Local Mean Difference", fontsize=26)
    plt.grid(True)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    
    # Increase font size for all text elements    plt.rcParams.update({'font.size': 22})
    
    plt.tight_layout()
    
    # Save both PDF and PNG versions
    plt.savefig(os.path.join(output_dir, 'global_local_vs_p2p.pdf'), 
                bbox_inches='tight', dpi=1000)
    plt.savefig(os.path.join(output_dir, 'global_local_vs_p2p.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()

def plot_spenders_caught_vs_ratio(results, output_dir, results_dir):
    """
    Plot relationship between ratio of double spenders to honest and number of spenders caught
    """
    plt.figure(figsize=(10, 7))
    
    # Collect data points for regression
    ratios = []
    spenders_caught = []
    
    for result in results:
        # Get the last value for spenders caught
        caught = result['spenders_caught_means'][-1]
        # Get the ratio parameter (index 3 in params)
        ratio = result['params'][3]
        
        ratios.append(ratio)
        spenders_caught.append(caught)
        
        actual_ratio_of_spenders_caught = [spenders_caught[idx] / (el / (1 + el)) for idx, el in enumerate(ratios)]

    plt.scatter(ratios, actual_ratio_of_spenders_caught, alpha=0.2, s=50, color='tab:blue')
    
    # Calculate regression line
    z = np.polyfit(ratios, actual_ratio_of_spenders_caught, 1)
    p = np.poly1d(z)
    
    # Plot regression line
    x_range = np.linspace(min(ratios), max(ratios), 100)
    plt.plot(x_range, p(x_range), "r--", alpha=0.8, linewidth=5.0)
    
    plt.xlabel("Malicious to Honest Ratio", fontsize=26)
    plt.ylabel("Ratio of Spenders Caught", fontsize=26)
    plt.grid(True)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    
    # Add vertical text label based on results directory name
    if results_dir:
        if 'urban' in results_dir.lower():
            area_type = 'Urban'
        elif 'rural' in results_dir.lower():
            area_type = 'Rural'
        else:
            area_type = None
            
        if area_type:
            # Get the current axis limits
            ax = plt.gca()
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            
            # Add vertical text on the right side
            plt.text(xlim[1] + (xlim[1] - xlim[0]) * 0.05,  # x position
                    (ylim[0] + ylim[1]) / 2,  # y position (middle of y-axis)
                    area_type,
                    rotation=90,  # vertical text
                    fontsize=26,
                    va='center',  # vertical alignment
                    ha='center')  # horizontal alignment
    
    # Increase font size for all text elements    plt.rcParams.update({'font.size': 22})
    
    plt.tight_layout()
    
    # Save both PDF and PNG versions
    plt.savefig(os.path.join(output_dir, 'spenders_caught_vs_ratio.pdf'), 
                bbox_inches='tight', dpi=1000)
    plt.savefig(os.path.join(output_dir, 'spenders_caught_vs_ratio.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Plot specific relationships between parameters and outputs')
    parser.add_argument('--results-dir', type=str, required=True,
                      help='Directory containing simulation results')
    parser.add_argument('--input-params-file', type=str, default='sobol_params.txt',
                      help='Input file containing parameter sets (default: sobol_params.txt)')
    
    args = parser.parse_args()
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                            'plots', f'plots_specific_relationships_{timestamp}')
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
    
    # Generate plots
    plot_counterfeit_vs_malicious(results, output_dir)
    print("Plotted counterfeit vs malicious")
    plot_global_local_vs_p2p(results, output_dir)
    print("Plotted global-local vs p2p")
    plot_spenders_caught_vs_ratio(results, output_dir, args.results_dir)
    print("Plotted spenders caught vs ratio")

if __name__ == "__main__":
    main() 

