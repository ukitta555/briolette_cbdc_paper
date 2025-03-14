import numpy as np

# Reading data from the file
def read_experiment_results(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # First two lines: arrays of integers (not plotted)
    array1 = list(map(int, lines[0].strip().split()))
    array2 = list(map(int, lines[1].strip().split()))

    # Third and fourth lines: single integers (not plotted)
    value1 = int(lines[2].strip())
    ratios_of_double_spent_coins = list(map(float, lines[3].strip().split()))
    ratio_of_double_spenders_caught = list(map(float, lines[4].strip().split()))


    # Fifth line: triplets of form "X Y Z", split by colon
    triplets = [tuple(map(float, triplet.split())) for triplet in lines[5].strip().split(':')][:-1]
    # Separate triplets into three arrays
    array_x = [triplet[0] for triplet in triplets]
    array_y = [triplet[1] for triplet in triplets]
    array_z = [triplet[2] for triplet in triplets]

    # Sixth line: array of floats
    float_array = list(map(float, lines[6].strip().split()))

    return array_x, array_y, array_z, float_array, ratios_of_double_spent_coins, ratio_of_double_spenders_caught

def read_experiment_results_avg_out(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    ratios_of_double_spent_coins = []
    ratio_of_double_spenders_caught = []

    for idx, line in enumerate(lines):
        # Convert each line into array of floats
        current_array = list(map(float, line.strip().split()))
        if idx % 2 == 0:
            ratios_of_double_spent_coins.append(current_array)
        else:
            ratio_of_double_spenders_caught.append(current_array)

    # print(ratios_of_double_spent_coins, ratio_of_double_spenders_caught)
    return ratios_of_double_spent_coins, ratio_of_double_spenders_caught