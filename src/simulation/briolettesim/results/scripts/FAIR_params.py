import numpy as np
from SALib.sample import sobol
from SALib import ProblemSpec

names_4 = ['move_prob', 'p2p', 'p2m', 'ds_ratio']
bounds_4 = [[0.05, 0.7],
            [0.1, 0.5],
            [0.1, 0.5],
            [0.05264, 3]]
bounds_4_realistic_adversary = [[0.05, 0.7],
            [0.1, 0.5],
            [0.1, 0.5],
            [0.05264, 3]]
bounds_bottom_10_4 = [[0.05332457115873695, 0.6976294308900832],
            [0.10273502804338933, 0.49859678372740746],
            [0.1007303286343813, 0.4989615138620138],
            [0.054364249609559774, 0.22308683157727122]]
bounds_top_10_4 = [[0.05313667901791633, 0.6997700172010809],
            [0.10036114566028119, 0.4984994933009148],
            [0.10007842518389226, 0.4978264268487692],
            [2.100510159525573, 2.999915667119175]]

problem = {
    'num_vars': 4,
    'names': names_4,
    'bounds': bounds_4_realistic_adversary
}

sp = ProblemSpec(problem)
sp.sample_sobol(1024, calc_second_order=False)

np.savetxt(f"results/FAIR_params_{problem["num_vars"]}/FAIR_params.txt", sp.samples, "%.17g")

# Old way
# param_values = sobol.sample(problem, 1024, calc_second_order=False)
# print(param_values.shape)
# np.savetxt("results/param_values.txt", param_values)

## manual way (uniform, not sobol)
# with open("results/FAIR_params_manual.txt", "w") as f:
#     for _ in range(100):
#         p2p = random.uniform(0.1, 0.5)
#         p2m = random.uniform(0.1, 0.5)
#         movement_prob = random.uniform(0.05, 0.7)
#         ds_ratio = random.uniform(0.05264, 3)
#         f.write(f"{p2p} {p2m} {movement_prob} {ds_ratio}\n")
#         # f.write(f"{x1:.6f} {x2:.6f} {x3:.6f} {x4:.6f}\n")
