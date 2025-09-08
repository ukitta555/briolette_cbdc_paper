import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import re
from datetime import datetime

from utils import read_experiment_results


def get_experiment_info(file_path):
    # Extract number of actors from filename
    # Looking for pattern like "50k" where the number represents thousands of actors
    actors_match = re.search(r'(\d+)k', file_path)
    if actors_match:
        num_actors = actors_match.group(1) + "k"
    else:
        num_actors = "unknown"
    
    # Determine if it's a predefined experiment
    is_predefined = 'predefined' in file_path.lower()
    experiment_type = 'predefined' if is_predefined else 'sobol'
    
    # Determine experiment category
    if 'threatlevel' in file_path.lower():
        category = 'threat_level'
    elif 'sync_params' in file_path.lower():
        category = 'sync_params'
    elif 'sobol' in file_path.lower():
        category = 'sobol'
    else:
        category = 'other'
    
    return num_actors, experiment_type, category


def plot_combined_threat_experiments(input_dir, output_dir):
    """Plot all threat experiments on one chart, excluding specific experiment IDs."""
    # Get all threat experiment files
    threat_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.txt') and 'threatlevel' in file.lower():
                # Extract experiment ID from filename
                exp_id_match = re.search(r'expid_(\d+)', file)
                if exp_id_match:
                    exp_id = int(exp_id_match.group(1))
                    # Skip specified experiment IDs
                    if exp_id not in [4, 5, 6, 11, 12, 13]:
                        threat_files.append(os.path.join(root, file))

    if not threat_files:
        print("No threat experiment files found!")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Define experiment labels and ratios
    experiment_info = {
        0: ("Urban", "1.0"),
        1: ("Urban", "0.5"),
        2: ("Urban", "0.25"),
        3: ("Urban", "0.1"),
        7: ("Rural", "1.0"),
        8: ("Rural", "0.5"),
        9: ("Rural", "0.25"),
        10: ("Rural", "0.1")
    }
    
    # Use a colormap for different experiments
    colors = plt.cm.viridis(np.linspace(0, 1, len(threat_files)))
    
    # Plot 1: Double Spenders Caught
    plt.figure(figsize=(10, 7))  # Smaller figure size for conference paper
    
    # Group files by area type and sort by threat level
    urban_files = []
    rural_files = []
    for file_path in threat_files:
        exp_id_match = re.search(r'expid_(\d+)', file_path)
        if exp_id_match:
            exp_id = int(exp_id_match.group(1))
            if exp_id in experiment_info:
                area_type, ratio = experiment_info[exp_id]
                if area_type == "Urban":
                    urban_files.append((file_path, float(ratio)))
                else:
                    rural_files.append((file_path, float(ratio)))
    
    # Sort files by threat level (ratio) in descending order
    urban_files.sort(key=lambda x: x[1], reverse=True)
    rural_files.sort(key=lambda x: x[1], reverse=True)
    
    # Plot urban files first
    for (file_path, _), color in zip(urban_files, plt.cm.viridis(np.linspace(0, 0.5, len(urban_files)))):
        exp_id_match = re.search(r'expid_(\d+)', file_path)
        if exp_id_match:
            exp_id = int(exp_id_match.group(1))
            if exp_id in experiment_info:
                area_type, ratio = experiment_info[exp_id]
                label = f'{area_type} {ratio}'
            else:
                label = f'Experiment {exp_id}'
        else:
            label = 'unknown'
        
        # Read experiment results
        _, _, _, _, ratio_of_double_spenders_caught, _, _, _, _ = read_experiment_results(file_path)
        
        # Plot the results
        x_values = list(range(len(ratio_of_double_spenders_caught)))
        plt.plot(x_values, ratio_of_double_spenders_caught, 
                marker='o', linestyle='-', color=color, 
                label=label, alpha=0.7, markersize=4)  # Smaller markers
    
    # Plot rural files
    for (file_path, _), color in zip(rural_files, plt.cm.viridis(np.linspace(0.5, 1, len(rural_files)))):
        exp_id_match = re.search(r'expid_(\d+)', file_path)
        if exp_id_match:
            exp_id = int(exp_id_match.group(1))
            if exp_id in experiment_info:
                area_type, ratio = experiment_info[exp_id]
                label = f'{area_type} {ratio}'
            else:
                label = f'Experiment {exp_id}'
        else:
            label = 'unknown'
        
        # Read experiment results
        _, _, _, _, ratio_of_double_spenders_caught, _, _, _, _ = read_experiment_results(file_path)
        
        # Plot the results
        x_values = list(range(len(ratio_of_double_spenders_caught)))
        plt.plot(x_values, ratio_of_double_spenders_caught, 
                marker='o', linestyle='-', color=color, 
                label=label, alpha=0.7, markersize=4)  # Smaller markers

    plt.xlabel("Simulation step (1 step = 1 hour)", fontsize=20)
    plt.ylabel("Ratio of double spenders caught", fontsize=20)
    plt.legend(loc='upper left', fontsize=16, bbox_to_anchor=(0.02, 0.98))
    plt.grid(True)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    
    # Save both PDF and PNG versions
    plt.savefig(os.path.join(output_dir, 'combined_threat_experiments_double_spenders.pdf'), 
                bbox_inches='tight', dpi=1000)
    plt.savefig(os.path.join(output_dir, 'combined_threat_experiments_double_spenders.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()

    # Plot 2: Counterfeit Ratio
    plt.figure(figsize=(10, 7))  # Smaller figure size for conference paper
    
    # Plot urban files first
    for (file_path, _), color in zip(urban_files, plt.cm.viridis(np.linspace(0, 0.5, len(urban_files)))):
        exp_id_match = re.search(r'expid_(\d+)', file_path)
        if exp_id_match:
            exp_id = int(exp_id_match.group(1))
            if exp_id in experiment_info:
                area_type, ratio = experiment_info[exp_id]
                label = f'{area_type} {ratio}'
            else:
                label = f'Experiment {exp_id}'
        else:
            label = 'unknown'
        
        # Read experiment results
        _, _, _, counterfeit_ratio, _, _, _, _, _ = read_experiment_results(file_path)
        
        # Plot the results
        x_values = list(range(len(counterfeit_ratio)))
        plt.plot(x_values, counterfeit_ratio, 
                marker='o', linestyle='-', color=color, 
                label=label, alpha=0.7, markersize=4)  # Smaller markers
    
    # Plot rural files
    for (file_path, _), color in zip(rural_files, plt.cm.viridis(np.linspace(0.5, 1, len(rural_files)))):
        exp_id_match = re.search(r'expid_(\d+)', file_path)
        if exp_id_match:
            exp_id = int(exp_id_match.group(1))
            if exp_id in experiment_info:
                area_type, ratio = experiment_info[exp_id]
                label = f'{area_type} {ratio}'
            else:
                label = f'Experiment {exp_id}'
        else:
            label = 'unknown'
        
        # Read experiment results
        _, _, _, counterfeit_ratio, _, _, _, _, _ = read_experiment_results(file_path)
        
        # Plot the results
        x_values = list(range(len(counterfeit_ratio)))
        plt.plot(x_values, counterfeit_ratio, 
                marker='o', linestyle='-', color=color, 
                label=label, alpha=0.7, markersize=4)  # Smaller markers

    plt.xlabel("Simulation step (1 step = 1 hour)", fontsize=20)
    plt.ylabel("Counterfeit ratio", fontsize=20)
    plt.legend(loc='upper left', fontsize=16, bbox_to_anchor=(0.02, 0.98))
    plt.grid(True)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    
    # Save both PDF and PNG versions
    plt.savefig(os.path.join(output_dir, 'combined_threat_experiments_counterfeit.pdf'), 
                bbox_inches='tight', dpi=1000)
    plt.savefig(os.path.join(output_dir, 'combined_threat_experiments_counterfeit.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()


def plot_experiment_results(file_path, output_base_dir, input_dir):
    # Extract experiment parameters from filename
    filename = os.path.basename(file_path)
    # Remove .txt extension
    filename = os.path.splitext(filename)[0]
    
    # Get experiment info
    num_actors, experiment_type, category = get_experiment_info(file_path)
    
    # Get relative path from the input directory to preserve structure
    rel_path = os.path.relpath(os.path.dirname(file_path), input_dir)
    
    # Create output directory preserving the structure and creating a folder for each experiment
    # Include both experiment_type and category in the path
    output_dir = os.path.join(output_base_dir, experiment_type, category, rel_path, filename)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing file: {file_path}")
    print(f"Output directory: {output_dir}")
    
    max_period_double_spent_seen_again, \
    max_transactions_double_spent_coin, \
    final_simulation_step, \
    counterfeit_ratio, \
    ratio_of_double_spenders_caught, \
    mean_global_local_diff, \
    std_global_local_diff, \
    max_global_local_diff, \
    std_intra_sample_diff = read_experiment_results(file_path)

    # Preprocess arrays to ensure they are monotonically increasing
    max_period_so_far = 0
    max_transactions_so_far = 0
    
    for i in range(len(max_period_double_spent_seen_again)):
        max_period_so_far = max(max_period_so_far, max_period_double_spent_seen_again[i][1])
        max_period_double_spent_seen_again[i] = (max_period_double_spent_seen_again[i][0], max_period_so_far)
        
    for i in range(len(max_transactions_double_spent_coin)):
        max_transactions_so_far = max(max_transactions_so_far, max_transactions_double_spent_coin[i][1])
        max_transactions_double_spent_coin[i] = (max_transactions_double_spent_coin[i][0], max_transactions_so_far)

    # Extract experiment ID and determine ratio
    exp_id_match = re.search(r'expid_(\d+)', filename)
    if exp_id_match:
        exp_id = int(exp_id_match.group(1))
        # Map experiment IDs to ratios
        ratio_map = {
            0: "1.0", 7: "1.0",
            1: "0.5", 8: "0.5",
            2: "0.25", 9: "0.25",
            3: "0.1", 10: "0.1"
        }
        ratio = ratio_map.get(exp_id, "Unknown")
        area_type = "Urban" if exp_id in [0, 1, 2, 3] else "Rural" if exp_id in [7, 8, 9, 10] else "Unknown"
    else:
        ratio = "Unknown"
        area_type = "Unknown"

    # Store the data for later plotting
    if not hasattr(plot_experiment_results, 'epoch_diff_data'):
        plot_experiment_results.epoch_diff_data = {}
    
    if ratio not in plot_experiment_results.epoch_diff_data:
        plot_experiment_results.epoch_diff_data[ratio] = {}
    
    plot_experiment_results.epoch_diff_data[ratio][area_type] = {
        'mean': mean_global_local_diff,
        'std': std_global_local_diff
    }

    # Plot the float array
    # plt.figure(figsize=(8, 5))  # Smaller figure size
    # x_values = list(range(len(std_intra_sample_diff)))
    # plt.plot(x_values, std_intra_sample_diff, marker='o', linestyle='-', color='r', label='Intra Std of Local Epochs', markersize=4)
    # plt.xlabel("Simulation step (1 step = 1 hour)", fontsize=18)
    # plt.ylabel("Measurement value", fontsize=18)
    # plt.legend(fontsize=14)
    # plt.grid(True)
    # plt.xticks(fontsize=14)
    # plt.yticks(fontsize=14)
    # plt.savefig(os.path.join(output_dir, 'intra_std_local_epochs.png'), dpi=300)
    # plt.close()

    # Plot the float array
    plt.figure(figsize=(8, 5))  # Smaller figure size
    x_values = list(range(len(counterfeit_ratio)))
    plt.plot(x_values, counterfeit_ratio, marker='o', linestyle='-', color='r', label='Counterfeit ratio', markersize=4)
    plt.xlabel("Simulation step (1 step = 1 hour)", fontsize=18)
    plt.ylabel("Counterfeit ratio", fontsize=18)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(os.path.join(output_dir, 'counterfeit_ratio.png'), dpi=300)
    plt.close()

    # Plot the float array
    plt.figure(figsize=(8, 5))  # Smaller figure size
    x_values = list(range(len(ratio_of_double_spenders_caught)))
    plt.plot(x_values, ratio_of_double_spenders_caught, marker='o', linestyle='-', color='r', label='Ratio of double spenders caught', markersize=4)
    plt.xlabel("Simulation step (1 step = 1 hour)", fontsize=18)
    plt.ylabel("Ratio of double spenders caught", fontsize=18)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(os.path.join(output_dir, 'double_spenders_caught.png'), dpi=300)
    plt.close()

    # Plot max period double spent seen again
    plt.figure(figsize=(8, 5))  # Smaller figure size
    steps = [x[0] for x in max_period_double_spent_seen_again]
    values = [x[1] for x in max_period_double_spent_seen_again]
    plt.plot(steps, values, marker='o', linestyle='-', color='b', label='Max Period Double Spent Seen Again', markersize=4)
    plt.xlabel("Step", fontsize=18)
    plt.ylabel("Period (hours)", fontsize=18)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(os.path.join(output_dir, 'max_period_double_spent.png'), dpi=300)
    plt.close()

    # Plot max transactions double spent coin
    plt.figure(figsize=(8, 5))  # Smaller figure size
    steps = [x[0] for x in max_transactions_double_spent_coin]
    values = [x[1] for x in max_transactions_double_spent_coin]
    plt.plot(steps, values, marker='o', linestyle='-', color='g', label='Max Transactions Double Spent Coin', markersize=4)
    plt.xlabel("Step", fontsize=18)
    plt.ylabel("Number of transactions", fontsize=18)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(os.path.join(output_dir, 'max_transactions_double_spent.png'), dpi=300)
    plt.close()


def process_directory(input_dir, output_dir):
    # First, determine the experiment type and number of actors
    experiment_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.txt'):
                experiment_files.append(os.path.join(root, file))
    
    if not experiment_files:
        print("No experiment files found!")
        return
    
    # Get info from the first file (assuming all files in a run have same parameters)
    num_actors, experiment_type, _ = get_experiment_info(experiment_files[0])
    
    # Create the output directory with descriptive name
    output_dir = os.path.join('plots', f"{num_actors}_actors_{experiment_type}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Process all files to collect data
    for file_path in experiment_files:
        plot_experiment_results(file_path, output_dir, input_dir)
    
    # Create combined epoch difference plots
    if hasattr(plot_experiment_results, 'epoch_diff_data'):
        for ratio in ['1.0', '0.5', '0.25', '0.1']:
            if ratio in plot_experiment_results.epoch_diff_data:
                plt.figure(figsize=(10, 6))  # Adjusted figure size
                data = plot_experiment_results.epoch_diff_data[ratio]
                
                # Plot Urban data
                if 'Urban' in data:
                    plt.plot(range(len(data['Urban']['mean'])), data['Urban']['mean'], 
                            marker='o', label='Urban Mean', markersize=4, alpha=0.5)
                    plt.plot(range(len(data['Urban']['std'])), data['Urban']['std'], 
                            marker='s', label='Urban Std', markersize=4, alpha=0.5)
                
                # Plot Rural data
                if 'Rural' in data:
                    plt.plot(range(len(data['Rural']['mean'])), data['Rural']['mean'], 
                            marker='^', label='Rural Mean', markersize=4, alpha=0.5)
                    plt.plot(range(len(data['Rural']['std'])), data['Rural']['std'], 
                            marker='v', label='Rural Std', markersize=4, alpha=0.5)
                
                plt.xlabel("Simulation step (1 step = 1 hour)", fontsize=20)
                plt.ylabel("Epoch difference", fontsize=20)
                plt.legend(fontsize=16, loc='upper right')
                plt.grid(True)
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'epoch_diff_ratio_{ratio}.pdf'), dpi=1000)
                plt.savefig(os.path.join(output_dir, f'epoch_diff_ratio_{ratio}.png'), 
                           bbox_inches='tight', dpi=300)
                plt.close()
    
    # Create combined threat experiment plot
    plot_combined_threat_experiments(input_dir, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate plots from experiment results.')
    parser.add_argument('input_dir', type=str, help='Directory containing experiment results files')
    args = parser.parse_args()
    
    print(f"Processing directory: {args.input_dir}")
    process_directory(args.input_dir, None)
