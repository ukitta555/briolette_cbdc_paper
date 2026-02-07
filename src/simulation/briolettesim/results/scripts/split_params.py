import os

target_dir = "results/FAIR_params_6"
original = open(os.path.join(target_dir, "FAIR_params.txt"))
n_slices = 32

params = original.readlines()
slice_size = int(len(params)/n_slices)

for file_idx in range(n_slices):
  output = open(os.path.join(target_dir, f"FAIR_params_{file_idx}.txt"), "w")
  output.writelines(params[file_idx:file_idx+slice_size])
  output.close()
