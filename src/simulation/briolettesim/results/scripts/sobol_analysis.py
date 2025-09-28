import numpy as np
import matplotlib.pyplot as plt
from SALib.analyze.sobol import analyze
import os
import seaborn as sns

def perform_sobol_analysis(results, output_dir=None, results_dir=None):
    """
    Perform Sobol sensitivity analysis on the results.
    
    Args:
        results: List of dictionaries containing experiment results
        output_dir: Directory to save the plots (optional)
        results_dir: Directory containing the results (used to determine Urban/Rural)
    """
    # Sort results by experiment ID
    results = sorted(results, key=lambda x: x['exp_id'])
    
    # Prepare data for Sobol analysis
    X = []  # Input parameters
    Y_double_spent = []  # Output: double spent ratio
    Y_spenders_caught = []  # Output: spenders caught ratio
    Y_epoch_diff_mean = []  # Output: epoch difference mean
    
    for result in results:
        if result['params'] is None:
            continue
            
        # Get the final values for outputs
        double_spent_final = result['double_spent_means'][-1] if result['double_spent_means'] else 0
        spenders_caught_final = result['spenders_caught_means'][-1] if result['spenders_caught_means'] else 0
        epoch_diff_mean_final = result['diff_epoch_mean_mean'][-1] if result['diff_epoch_mean_mean'] else 0
        
        # Convert malicious-to-honest ratio to malicious-to-total ratio for the fourth parameter
        params = result['params'].copy()
        params[3] = params[3] / (1 + params[3])  # Convert malicious/honest to malicious/total
        
        X.append(params)
        Y_double_spent.append(double_spent_final)
        Y_spenders_caught.append(spenders_caught_final)
        Y_epoch_diff_mean.append(epoch_diff_mean_final)
    
    X = np.array(X)
    Y_double_spent = np.array(Y_double_spent)
    Y_spenders_caught = np.array(Y_spenders_caught)
    Y_epoch_diff_mean = np.array(Y_epoch_diff_mean)
    
    # Define the problem for Sobol analysis
    problem = {
        'num_vars': 4,
        'names': [
            'Move', 
            'P2P', 
            'P2M',
            'DS_Ratio'
        ],
        'bounds': [
            [0.05, 0.7],  # move_prob
            [0.1, 0.5],   # p2p_prob
            [0.1, 0.5],   # p2m_prob
            [0.091, 0.875]  # ratio_double_spenders_to_total (converted from [0.1, 7] malicious/honest)
        ],
        "outputs": [
            "ratio_of_double_spent_coins", 
            "ratio_of_double_spenders_caught",
            "global_to_local_epoch_diffs_mean"
        ]
    }
    
    # Perform Sobol analysis for each output
    print("\n=== Sobol Sensitivity Analysis ===")
    
    # Double spent ratio
    print("\n--- Double Spent Ratio ---")
    sobol_indices_double_spent = analyze(problem, Y_double_spent, calc_second_order=True)
    print_sobol_results(sobol_indices_double_spent, problem['names'])
    
    # Spenders caught ratio
    print("\n--- Spenders Caught Ratio ---")
    sobol_indices_spenders_caught = analyze(problem, Y_spenders_caught, calc_second_order=True)
    print_sobol_results(sobol_indices_spenders_caught, problem['names'])
    
    # Epoch difference mean
    print("\n--- Epoch Difference Mean ---")
    sobol_indices_epoch_diff = analyze(problem, Y_epoch_diff_mean, calc_second_order=True)
    print_sobol_results(sobol_indices_epoch_diff, problem['names'])
    
    # Plot first and total order indices
    plot_sobol_indices(sobol_indices_double_spent, sobol_indices_spenders_caught, 
                      sobol_indices_epoch_diff, problem['names'], output_dir, results_dir)
    
    # Plot second order indices
    plot_second_order_indices(sobol_indices_double_spent, sobol_indices_spenders_caught,
                            sobol_indices_epoch_diff, problem['names'], output_dir, results_dir)


def print_sobol_results(sobol_indices, param_names):
    """
    Print Sobol sensitivity indices in a readable format.
    
    Args:
        sobol_indices: Dictionary containing Sobol indices
        param_names: List of parameter names
    """
    print("\nFirst-order indices:")
    for i, name in enumerate(param_names):
        print(f"{name}: {sobol_indices['S1'][i]:.4f} ± {sobol_indices['S1_conf'][i]:.4f}")
    
    print("\nTotal-order indices:")
    for i, name in enumerate(param_names):
        print(f"{name}: {sobol_indices['ST'][i]:.4f} ± {sobol_indices['ST_conf'][i]:.4f}")
    
    print("\nSecond-order indices:")
    for i, name_i in enumerate(param_names):
        for j, name_j in enumerate(param_names[i+1:], i+1):
            print(f"{name_i}-{name_j}: {sobol_indices['S2'][i][j]:.4f} ± {sobol_indices['S2_conf'][i][j]:.4f}")


def plot_sobol_indices(sobol_indices_ds, sobol_indices_sc, sobol_indices_ed, param_names, output_dir=None, results_dir=None):
    """
    Plot Sobol sensitivity indices for all outputs.
    
    Args:
        sobol_indices_ds: Sobol indices for double spent ratio
        sobol_indices_sc: Sobol indices for spenders caught ratio
        sobol_indices_ed: Sobol indices for epoch difference mean
        param_names: List of parameter names
        output_dir: Directory to save the plots (optional)
        results_dir: Directory containing the results (used to determine Urban/Rural)
    """
    # Create three separate figures
    outputs = [
        ('Double Spent Ratio', sobol_indices_ds),
        ('Spenders Caught Ratio', sobol_indices_sc),
        ('Epoch Difference Mean', sobol_indices_ed)
    ]
    
    for i, (title, sobol_indices) in enumerate(outputs):
        plt.figure(figsize=(8, 6))
        
        # Plot first-order indices
        x = np.arange(len(param_names))
        width = 0.35
        spacing = 0.1  # Space between first-order and total-order bars
        
        rects1 = plt.bar(x - width/2 - spacing/2, sobol_indices['S1'], width, 
                        label='First-order', yerr=sobol_indices['S1_conf'], capsize=5)
        rects2 = plt.bar(x + width/2 + spacing/2, sobol_indices['ST'], width, 
                        label='Total-order', yerr=sobol_indices['ST_conf'], capsize=5)
        
        # Only add ylabel for the first plot
        if i == 0:
            plt.ylabel('Sensitivity Index', fontsize=20)
        plt.xticks(x, param_names, rotation=20, ha='right', fontsize=16)
        plt.legend(fontsize=18)
        plt.tick_params(axis='y', labelsize=16)
        
        # Add vertical text label only for the last figure (Epoch Difference Mean)
        if i == 2 and results_dir:
            if 'urban' in results_dir.lower():
                area_type = 'Urban'
            elif 'rural' in results_dir.lower():
                area_type = 'Rural'
            else:
                area_type = None
                
            if area_type:
                xlim = plt.gca().get_xlim()
                ylim = plt.gca().get_ylim()
                
                # Add vertical text on the right side
                plt.text(xlim[1] + (xlim[1] - xlim[0]) * 0.05,  # x position
                        (ylim[0] + ylim[1]) / 2,  # y position (middle of y-axis)
                        area_type,
                        rotation=90,  # vertical text
                        fontsize=26,
                        va='center',  # vertical alignment
                        ha='center')  # horizontal alignment
        
        plt.tight_layout()
        
        # Save the plot
        if output_dir:
            plt.savefig(os.path.join(output_dir, f'sobol_sensitivity_analysis_{i+1}.pdf'), 
                       bbox_inches='tight', dpi=1000)
            plt.savefig(os.path.join(output_dir, f'sobol_sensitivity_analysis_{i+1}.png'), 
                       bbox_inches='tight', dpi=300)
        else:
            plt.savefig(f'sobol_sensitivity_analysis_{i+1}.pdf', bbox_inches='tight', dpi=1000)
            plt.savefig(f'sobol_sensitivity_analysis_{i+1}.png', bbox_inches='tight', dpi=300)
        plt.close()


def plot_second_order_indices(sobol_indices_ds, sobol_indices_sc, sobol_indices_ed, param_names, output_dir=None, results_dir=None):
    """
    Plot second-order Sobol indices as heatmaps for all outputs.
    
    Args:
        sobol_indices_ds: Sobol indices for double spent ratio
        sobol_indices_sc: Sobol indices for spenders caught ratio
        sobol_indices_ed: Sobol indices for epoch difference mean
        param_names: List of parameter names
        output_dir: Directory to save the plots (optional)
        results_dir: Directory containing the results (used to determine Urban/Rural)
    """
    # Function to create second-order matrix and confidence matrix
    def create_second_order_matrices(sobol_indices):
        n = len(param_names)
        matrix = np.zeros((n, n))
        conf_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                matrix[i, j] = sobol_indices['S2'][i, j]
                matrix[j, i] = matrix[i, j]  # Symmetric
                conf_matrix[i, j] = sobol_indices['S2_conf'][i, j]
                conf_matrix[j, i] = conf_matrix[i, j]  # Symmetric
        return matrix, conf_matrix
    
    # Function to format annotation with confidence interval
    def format_annotation(val, conf):
        return f"{val:.3f}\n±{conf:.3f}"
    
    # Create three separate figures
    outputs = [
        ('Double Spent Ratio', sobol_indices_ds),
        ('Spenders Caught Ratio', sobol_indices_sc),
        ('Epoch Difference Mean', sobol_indices_ed)
    ]
    
    for i, (title, sobol_indices) in enumerate(outputs):
        plt.figure(figsize=(10, 8))
        
        # Create matrices for the output
        matrix, conf = create_second_order_matrices(sobol_indices)
        
        # Create annotation matrix
        annot = np.array([[format_annotation(matrix[i,j], conf[i,j]) 
                          for j in range(len(param_names))] 
                         for i in range(len(param_names))])
        
        # Plot heatmap
        sns.heatmap(matrix, annot=annot, fmt='', cmap='RdBu_r', 
                    xticklabels=param_names, yticklabels=param_names,
                    cbar_kws={'pad': 0.02},
                    annot_kws={'size': 20})
        
        plt.tick_params(axis='both', which='major', labelsize=20)
        plt.gca().collections[0].colorbar.ax.tick_params(labelsize=20)
        
        # Add vertical text label only for the last figure (Epoch Difference Mean)
        if i == 2 and results_dir:
            if 'urban' in results_dir.lower():
                area_type = 'Urban'
            elif 'rural' in results_dir.lower():
                area_type = 'Rural'
            else:
                area_type = None
                
            if area_type:
                xlim = plt.gca().get_xlim()
                ylim = plt.gca().get_ylim()
                
                # Add vertical text on the right side, moved further right to avoid colorbar
                plt.text(xlim[1] + (xlim[1] - xlim[0]) * 0.35,  # x position
                        (ylim[0] + ylim[1]) / 2,  # y position (middle of y-axis)
                        area_type,
                        rotation=90,  # vertical text
                        fontsize=26,
                        va='center',  # vertical alignment
                        ha='center')  # horizontal alignment
        
        plt.tight_layout()
        
        # Save both PDF and PNG versions
        if output_dir:
            plt.savefig(os.path.join(output_dir, f'sobol_second_order_analysis_{i+1}.pdf'), 
                       bbox_inches='tight', dpi=1000)
            plt.savefig(os.path.join(output_dir, f'sobol_second_order_analysis_{i+1}.png'), 
                       bbox_inches='tight', dpi=300)
        else:
            plt.savefig(f'sobol_second_order_analysis_{i+1}.pdf', bbox_inches='tight', dpi=1000)
            plt.savefig(f'sobol_second_order_analysis_{i+1}.png', bbox_inches='tight', dpi=300)
        plt.close() 