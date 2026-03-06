import os

target_dir = "results/FAIR_params_6"
n_slices = 32

original = open(os.path.join(target_dir, "FAIR_params.txt"))
params = original.readlines()
original.close()

slice_size = int(len(params)/n_slices)

for file_idx in range(n_slices):
  start = file_idx*slice_size  
  output = open(os.path.join(target_dir, f"FAIR_params_{file_idx}.txt"), "w")
  output.writelines(params[start:start+slice_size])
  output.close()
