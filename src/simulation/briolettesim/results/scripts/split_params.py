import os

target_dir = "results/FAIR_params_4_realistic"
original = open(os.path.join(target_dir, "FAIR_params.txt"))

params = original.readlines()
slice_size = int(len(params)/16)

for file_idx in range(16):
  output = open(os.path.join(target_dir, f"FAIR_params_{file_idx}.txt"), "w")
  output.writelines(params[file_idx:file_idx+slice_size])
  output.close()
