import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import beta as beta_func
from scipy.stats import beta as beta_dist

# Directory containing the files

# manual (uniform)
# target_dir = "results/sobol_experiments/experiment_20251031_024149_1k_actors/urban"

# sobol (swapped params)
# target_dir = "results/sobol_experiments/experiment_20251111_023326_1k_actors/urban"

# sobol (FAIR urban 1k)
# target_dir = "results/sobol_experiments/experiment_20251212_012438_1k_actors/urban"

# sobol (FAIR urban 10k)
target_dir = "results/sobol_experiments/FAIR_10k_urban"
# target_dir = "results/sobol_experiments/FAIR_10k_rural"

CoF, PL, TTXS = [], [], []
CoF_dict = {}
# target_dir = "results/sobol_experiments/FAIR_10k_urban_realistic"
# target_dir = "results/sobol_experiments/FAIR_10k_urban_1-5k_1-50"
# target_dir = "results/sobol_experiments/FAIR_10k_urban_1-1k_1-10"
# target_dir = "results/sobol_experiments/FAIR_10k_urban_4-params_2-10k_2-100"
# target_dir = "results/sobol_experiments/FAIR_10k_urban_6-params_2-10k_2-100"
# target_dir = "results/sobol_experiments/FAIR_10k_urban_4-params_1-1k_1-10"
# target_dir = "results/sobol_experiments/original_test"
# target_dir = "results/sobol_experiments/FAIR_10k_urban_0.75_test"
# target_dir = "results/sobol_experiments/FAIR_10k_urban_percent-values"
# target_dir = "results/sobol_experiments/FAIR_10k_urban_0.002-3_test"
# target_dir = "results/sobol_experiments/FAIR_10k_urban_0.002-1_test"
# target_dir = "results/sobol_experiments/FAIR_10k_urban_0.002-1_double_epoch_test" (PL: use 1200 bin size)
# target_dir = "results/sobol_experiments/FAIR_10k_urban_0.002-1_16_epoch_tickets_test"
# target_dir = "results/sobol_experiments/FAIR_10k_urban_0.002-1_20_epoch_tickets_0_lowerlimit_test"
# target_dir = "results/sobol_experiments/FAIR_10k_urban_0.002-1_20_epoch_tickets_0_lowerlimit_larger_test"
# target_dir = "results/sobol_experiments/FAIR_10k_urban_0.02-1_20_epoch_tickets_0_lowerlimit_larger_test"
# target_dir = "results/sobol_experiments/FAIR_10k_urban_0.02-1_20_0_200"
# target_dir = "results/sobol_experiments/FAIR_10k_urban_0.02-1_6_0_200"
# target_dir = "results/sobol_experiments/FAIR_10k_urban_0.02-1_12_0_200"


def extract_parameter_values(filename):
    """
    Extract the p2p, p2m, ratio_ds, p_mov values from a filename formatted as:
    model_Urban/Rural_p2p_x_p2m_x_ratiodoublespenders_x_move_x_expid_x_Sobol_xk_actors.txt
    """
    parts = filename.split("_")

    try:
        p2p = float(parts[-12])
        p2m = float(parts[-10])
        ratio_ds = float(parts[-8])
        p_mov = float(parts[-6])
        return [p_mov, p2p, p2m, ratio_ds]
    except:
        return None

def compute_bounds(data):
    sorted_keys = sorted(data.keys())
    N = max(1, int(0.10 * len(sorted_keys)))

    bottom_keys = sorted_keys[:N]
    top_keys = sorted_keys[-N:]

    subset_dict_bottom = {k: data[k] for k in bottom_keys}
    subset_dict_top = {k: data[k] for k in top_keys}

    for i,subset_dict in enumerate([subset_dict_bottom, subset_dict_top]):
        # compute bounds for p2p,p2m,ratio_ds,p_move within this subset
        subset_values = np.array(list(subset_dict.values()))  # shape (N, 4)
        p_move_vals = subset_values[:, 0]
        p2p_vals = subset_values[:, 1]
        p2m_vals = subset_values[:, 2]
        ratio_ds_vals = subset_values[:, 3]

        p_move_bounds = (float(np.min(p_move_vals)), float(np.max(p_move_vals)))
        p2p_bounds = (float(np.min(p2p_vals)), float(np.max(p2p_vals)))
        p2m_bounds = (float(np.min(p2m_vals)), float(np.max(p2m_vals)))
        ratio_ds_bounds = (float(np.min(ratio_ds_vals)), float(np.max(ratio_ds_vals)))

        print(f"\nBounds for {["bottom","top"][i]} 10% (p_move,p2p,p2m,ratio_ds):")
        print("p_move:", p_move_bounds)
        print("p2p:", p2p_bounds)
        print("p2m:", p2m_bounds)
        print("ratio_ds:", ratio_ds_bounds)

def PERT(x, min, mode, max, gamma=4):
    """PERT function with min, mode, and max"""

    alpha = (1.0 + (gamma * ((mode-min) / (max-min))))
    beta = (1.0 + (gamma * ((max-mode) / (max-min))))
    #print(x)
    #print(alpha,beta)
    # one = (x-min)**(alpha-1)
    # #print(beta-1)
    # two = (max-x)**(beta-1)
    # #print(two)
    # three = beta_func(alpha,beta)
    # four = (max-min)**(alpha+beta-1)
    # return (one*two)/(three*four)

    return beta_dist.pdf(x, alpha, beta, min, max-min)

def modPertModel(x, a, b, c, scale=1, locY=0):
    """Manual PERT function with min, mode, max, and scaling factors scale, locY"""

    alphaParam = 1.0 + (4.0 * (b - a) / (c - a))
    betaParam = 1.0 + (4.0 * (c - b) / (c - a))
    betaFcnSample = beta_func(alphaParam, betaParam)

    firstNumer = np.sign(x - a) * (np.abs(x - a) ** (alphaParam - 1))
    secondNumer = np.sign(c - x) * (np.abs(c - x) ** (betaParam - 1))
    secondDenom = np.sign(c - a) * (np.abs(c - a) ** (alphaParam + betaParam - 1))
    numer = firstNumer * secondNumer
    denom = betaFcnSample * secondDenom

    return ((numer / denom) * scale) + locY
    #return (beta_dist.pdf(x, alphaParam, betaParam, a, c-a) * scale ) + locY

def fit_curve(x_data, y_data, initial_guess, use_built_in_model, fit_gamma):
    if use_built_in_model:
        # bounds=([0,0,0,1], [np.inf,np.inf,np.inf,10])
        # bounds=([initial_guess[0],0,0,1], [np.inf,np.inf,np.inf,10])
        bounds=([-np.inf,-np.inf,-np.inf,1], [np.inf,np.inf,np.inf,10])
        if fit_gamma:
            # Fit with gamma
            param_names = ['Minimum', 'Mode', 'Maximum', 'Gamma']
            popt, pcov, infodict, mseg, ier = curve_fit(PERT, x_data, y_data, p0=initial_guess, bounds=bounds, full_output=True)
        else:
            # Fit without gamma
            param_names = ['Minimum', 'Mode', 'Maximum']
            popt, pcov, infodict, mseg, ier = curve_fit(PERT, x_data, y_data, p0=initial_guess[:-1], bounds= (bounds[0][:-1], bounds[1][:-1]), full_output=True)
    else:
        param_names = ['Minimum', 'Mode', 'Maximum', 'Scale', 'LocY']
        popt, pcov, infodict, mseg, ier = curve_fit(modPertModel, x_data, y_data, p0=initial_guess, full_output=True, maxfev=48000)

    # Calculate confidence intervals
    perr = np.sqrt(np.diag(pcov))
    for param, err, name in zip(popt, perr, param_names):
        print(f"{name}: {param:.2f} ± {err:.2f}")
    #print(infodict,mseg,ier)
    return popt

def build_hist(data, bin_width):
    """Function to build histogram from a list of floats"""
    if not data:
        return []
    min_val = min(data)
    max_val = max(data)
    bins = []
    current = min_val
    while current < max_val:
        upper = current + bin_width
        count = sum(current <= x < upper for x in data)
        bins.append((current, upper, count))
        current = upper
    return bins

def fit_and_plot_hist(ax, hist, label, bin_width):
    """Function to fit the curves and plot histograms"""
    use_built_in_model =  False
    fit_gamma = False

    if not hist:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        return

    bin_starts = [b[0] for b in hist]
    counts = [b[2] for b in hist]

    # Use bin centers for x_data
    x_data = np.array([b+bin_width/2 for b in bin_starts])
    counts = np.array(counts, dtype=np.float64)

    # Convert counts to probability density
    total_area = np.sum(counts) * bin_width
    prob_density = counts / total_area

    # Normalize data
    # x_data = (x_data - np.min(x_data))/(np.max(x_data) - np.min(x_data))
    # counts = (counts - np.min(counts))/(np.max(counts) - np.min(counts))

    # Initial guess: min, mode (peak), max
    mode_idx = np.argmax(counts)
    if use_built_in_model:
        init_guess = [np.min(x_data), x_data[mode_idx], np.max(x_data), 4]
    else:
        init_guess = [np.min(x_data), x_data[mode_idx], np.max(x_data), 2000, 0]

    print(f"\n{label} - Initial guess: {init_guess}")

    # Fit the curve to probability density
    popt = fit_curve(x_data,prob_density,init_guess, use_built_in_model, fit_gamma)

    ax.bar(bin_starts, prob_density, width=bin_width, align='edge') #, label='Data')
    if use_built_in_model:
        ax.plot(x_data, PERT(x_data,*popt), 'r-') #, label='PERT fit')
        #ax.plot(x_data[:-7], modPertModel(x_data[:-7],*popt), 'g+') #, label='PERT fit')
        #diff = PERT(x_data,*popt) - modPertModel(x_data,*popt)
        #print(diff)
    else:
        # t = popt.copy()
        # t[-1] = 0
        #ax.plot(x_data, modPertModel(x_data,*popt), 'g+') #, label='PERT fit')
        #print(*popt)
        ax.plot(x_data, (PERT(x_data,*(popt[:-2]))* popt[-2]) + popt[-1], 'r-', linewidth=1) #, label='PERT fit')
        #ax.axvline(x=popt[0], color='black', linestyle='--')
        #ax.axvline(x=popt[2], color='black', linestyle='--')

    ax.set_title(label)
    # ax.set_xticks([10000,20000,30000,40000,50000,60000,70000,80000,90000,100000]) # CoF
    # ax.set_xticks([25000,50000,75000,100000,125000,150000,175000,200000,225000,250000,275000,300000]) # PL
    ax.set_xlabel(x_labels[label])
    ax.set_ylabel("Probability")
    ax.set_axisbelow(True)
    # ax.legend()
    ax.grid(True, alpha=0.3, linestyle='dashdot', linewidth=0.5)

### Program starts here ###
single_graph = False
average_values = False
diagrams_to_draw = ["Contact Frequency"] #["Contact Frequency", "Primary Loss", "Total Transactions"]

# Iterate over all files in the target directory
for filename in os.listdir(target_dir):
  filepath = os.path.join(target_dir, filename)
  if not os.path.isfile(filepath):
    continue  # skip subdirectories or non-files

#   params = extract_parameter_values(filename)
#   if params is None:
#       continue

  # Parse the file line by line
  with open(filepath, "r") as f:
    if average_values:
      tmp_cof= []
      tmp_pl= []
      tmp_ttxs= []
    
    for line in f:
      parts = line.strip().split()
      if len(parts) != 3:
        continue  # skip malformed lines
      cof, pl, ttxs = map(float, parts)
      
      if average_values:
        tmp_cof.append(cof)
        tmp_pl.append(pl)
        tmp_ttxs.append(ttxs)
      else:
        CoF.append(cof)
        PL.append(pl)
        TTXS.append(ttxs)
    
    #   CoF_dict[cof] = params
    
    if average_values:
      CoF.append(np.mean(tmp_cof))
      PL.append(np.mean(tmp_pl))
      TTXS.append(np.mean(ttxs))

# Compute bounds
# compute_bounds(CoF_dict)

# Build histograms
hist_CoF = build_hist(CoF, 500)
# hist_PL = build_hist(PL, 1500)
# hist_TTXS = build_hist(TTXS, 1500)

# # Print results for each file
# print(f"\nFile: {filename}")
# print("CoF histogram (bin_start, bin_end, count):")
# for b in hist_CoF:
#     print(b)
# print("PL histogram (bin_start, bin_end, count):")
# for b in hist_PL:
#     print(b)
# print("TTXS histogram (bin_start, bin_end, count):")
# for b in hist_TTXS:
#     print(b)

# Plot histograms using matplotlib
x_labels = {"Contact Frequency":"Double spend transactions", "Primary Loss":"Total double spent value", "Total Transactions":"Total transactions"}
if single_graph:
    fig, axes = plt.subplots(3, 1, figsize=(8, 10))
    fig.suptitle(f"Histograms for {':'.join(target_dir.split('/')[-2:])}")
else:
    figs = []
    axes = []
    for label in diagrams_to_draw:
        fig = plt.figure(figsize=(10, 4))
        ax = plt.gca()
        ax.set_title(f"{label} — {':'.join(target_dir.split('/')[-2:])}")
        figs.append(fig)
        axes.append(ax)


fit_and_plot_hist(axes[0], hist_CoF, "Contact Frequency", 500)
# fit_and_plot_hist(axes[0], hist_PL, "Primary Loss", 1500)
# fit_and_plot_hist(axes[2], hist_TTXS, "Total Transactions", 1500)

if single_graph:
    plt.tight_layout()
    #plt.show()
    plt.savefig("results/plots/FAIR_hist.png")
    plt.close()
else:
    for fig, name in zip(figs, diagrams_to_draw):
        fig.tight_layout()
        fig.savefig(f"results/plots/{target_dir.split("/")[-1]}_hist_{name}.pdf")
        plt.close(fig)
