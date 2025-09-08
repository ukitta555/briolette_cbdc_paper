import numpy as np

# Reading data from the file
def read_experiment_results(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # First two lines: arrays of (epoch, value) pairs
    max_period_double_spent_seen_again = []
    max_transactions_double_spent_coin = []
    
    # Parse first line: epoch and life measurements
    # values = lines[0].strip().split()
    # for i in range(0, len(values), 2):
    #     epoch = int(values[i])
    #     life = int(values[i + 1])
    #     max_period_double_spent_seen_again.append((epoch, life))
    
    # # Parse second line: epoch and transaction measurements
    # values = lines[1].strip().split()
    # for i in range(0, len(values), 2):
    #     epoch = int(values[i])
    #     txs = int(values[i + 1])
    #     max_transactions_double_spent_coin.append((epoch, txs))

    # Third and fourth lines: single integers (not plotted)
    # final_simulation_step = int(lines[2].strip())
    final_simulation_step = None
    counterfeit_ratio = list(map(float, lines[0].strip().split()))
    ratio_of_double_spenders_caught = list(map(float, lines[1].strip().split()))

    # Fifth line: triplets of form "X Y Z", split by colon
    triplets = [tuple(map(float, triplet.split())) for triplet in lines[2].strip().split(':')][:-1]
    # Separate triplets into three arrays
    mean_global_local_diff = [triplet[0] for triplet in triplets]
    std_global_local_diff = [triplet[1] for triplet in triplets]
    max_global_local_diff = [triplet[2] for triplet in triplets]

    # Sixth line: array of floats
    # std_intra_sample_diff = list(map(float, lines[6].strip().split()))

    std_intra_sample_diff = None

    return max_period_double_spent_seen_again, \
        max_transactions_double_spent_coin, \
        final_simulation_step, \
        counterfeit_ratio, \
        ratio_of_double_spenders_caught, \
        mean_global_local_diff, \
        std_global_local_diff, \
        max_global_local_diff, \
        std_intra_sample_diff

def read_experiment_results_avg_out(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    ratios_of_double_spent_coins = []
    ratio_of_double_spenders_caught = []
    global_to_local_epoch_diffs_mean = []
    global_to_local_epoch_diffs_std = []
    global_to_local_epoch_diffs_max = []

    for idx, line in enumerate(lines):
        if idx % 3 == 0:
            # Convert line into array of floats
            current_array = list(map(float, line.strip().split()))
            ratios_of_double_spent_coins.append(current_array)
        elif idx % 3 == 1:
            # Convert line into array of floats
            current_array = list(map(float, line.strip().split()))
            ratio_of_double_spenders_caught.append(current_array)
        elif idx % 3 == 2:
            # Convert line  into array of triplets
            triplets = [tuple(map(float, triplet.split())) for triplet in line.strip().split(':')][:-1]
            global_to_local_epoch_diffs_mean.append([triplet[0] for triplet in triplets])
            global_to_local_epoch_diffs_std.append([triplet[1] for triplet in triplets])
            global_to_local_epoch_diffs_max.append([triplet[2] for triplet in triplets])


    # print(ratios_of_double_spent_coins, ratio_of_double_spenders_caught)
    return ratios_of_double_spent_coins, \
        ratio_of_double_spenders_caught, \
        global_to_local_epoch_diffs_mean, \
        global_to_local_epoch_diffs_std, \
        global_to_local_epoch_diffs_max

