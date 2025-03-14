import numpy as np
import matplotlib.pyplot as plt

# Read the data from sobol_params.txt
data = np.loadtxt('sobol_params.txt')

print(data)


new_data_x = []
new_data_y = []
for i in range(0, data.shape[0]):
    new_data_x.append(data[i][0])
    new_data_y.append(data[i][1])

print(len(new_data_x), len(new_data_y))
print("new_data_x: ", new_data_x)
print("new_data_y: ", new_data_y)


# Create scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(new_data_x, new_data_y, alpha=0.6)
plt.xlabel('Parameter Value 1')
plt.ylabel('Parameter Value 2')
plt.title('Sobol Parameters Distribution')
plt.grid(True, alpha=0.3)

# Save the plot
plt.savefig('sobol_params_scatter.png')
plt.close()
