
from matplotlib import pyplot as plt


def plot_results_p2p_p2m(results):
    plt.figure(figsize=(25, 15))
    
    # Plot 3: Double spent ratio vs ratio_double_spenders_to_honest
    plt.subplot(2, 4, 1)
    x = [result['params'][0] for result in results]
    y = [result['double_spent_means'][-1] for result in results]
    plt.scatter(x, y)
    plt.title('Final Double Spent Ratio vs P2P probability')
    plt.xlabel('P2P probability')
    plt.ylabel('Final Double Spent Ratio')
    
    # Plot 4: Double spent ratio vs random_sync_probability
    plt.subplot(2, 4, 2)
    x = [result['params'][1] for result in results]
    y = [result['double_spent_means'][-1] for result in results]
    plt.scatter(x, y)
    plt.title('Final Double Spent Ratio vs P2M probability')
    plt.xlabel('P2M probability')
    plt.ylabel('Final Double Spent Ratio')
    
    # Plot 5: Spenders caught vs ratio_double_spenders_to_honest
    plt.subplot(2, 4, 3)
    x = [result['params'][0] for result in results]
    y = [result['diff_epoch_mean_mean'][-1] for result in results]
    plt.scatter(x, y)
    plt.title('Final epoch diff (mean of means) vs P2P probability')
    plt.xlabel('P2P probability')
    plt.ylabel('Final epoch diff (mean of means)')
    
    # Plot 6: Spenders caught vs random_sync_probability
    plt.subplot(2, 4, 4)
    x = [result['params'][1] for result in results]
    y = [result['diff_epoch_mean_mean'][-1] for result in results]
    plt.scatter(x, y)
    plt.title('Final epoch diff (mean of means) vs P2M probability')
    plt.xlabel('P2M probability')
    plt.ylabel('Final epoch diff (mean of means)')

     # Plot 6: Spenders caught vs random_sync_probability
    plt.subplot(2, 4, 5)
    x = [result['params'][0] for result in results]
    y = [result['diff_epoch_max_mean'][-1] for result in results]
    plt.scatter(x, y)
    plt.title('Final epoch diff (mean of max) vs P2P probability')
    plt.xlabel('P2P probability')
    plt.ylabel('Final epoch diff (mean of max)')

     # Plot 6: Spenders caught vs random_sync_probability
    plt.subplot(2, 4, 6)
    x = [result['params'][1] for result in results]
    y = [result['diff_epoch_max_mean'][-1] for result in results]
    plt.scatter(x, y)
    plt.title('Final epoch diff (mean of max) vs P2M probability')
    plt.xlabel('P2M probability')
    plt.ylabel('Final epoch diff (mean of max)')


     # Plot 3: Double spent ratio vs ratio_double_spenders_to_honest
    plt.subplot(2, 4, 7)
    x = [result['params'][0] for result in results]
    y = [result['spenders_caught_means'][-1] for result in results]
    plt.scatter(x, y)
    plt.title('Spenders caught ratio vs P2P probability')
    plt.xlabel('P2P probability')
    plt.ylabel('Spenders caught ratio')
    
    # Plot 4: Double spent ratio vs random_sync_probability
    plt.subplot(2, 4, 8)
    x = [result['params'][1] for result in results]
    y = [result['spenders_caught_means'][-1] for result in results]
    plt.scatter(x, y)
    plt.title('Spenders caught ratio vs P2M probability')
    plt.xlabel('P2M probability')
    plt.ylabel('Spenders caught ratio')

    
    plt.tight_layout()
    plt.savefig('experiment_results_p2p_p2m.png')
    plt.close()
