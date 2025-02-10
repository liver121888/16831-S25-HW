import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../output')


# File paths
ant_avg_file = os.path.join(data_path, "avg_q2_dagger_ant_Ant-v2_10-02-2025_00-52-51.csv")
ant_std_file = os.path.join(data_path, "std_q2_dagger_ant_Ant-v2_10-02-2025_00-52-51.csv")
humanoid_avg_file = os.path.join(data_path, "avg_q2_dagger_humanoid_Humanoid-v2_09-02-2025_23-54-04.csv")
humanoid_std_file = os.path.join(data_path, "std_q2_dagger_humanoid_Humanoid-v2_09-02-2025_23-54-04.csv")

# Load data
ant_avg = pd.read_csv(ant_avg_file)
ant_std = pd.read_csv(ant_std_file)
humanoid_avg = pd.read_csv(humanoid_avg_file)
humanoid_std = pd.read_csv(humanoid_std_file)

# Extract relevant columns
x_axis_ant = ant_avg.iloc[:, 1]  # DAgger iterations
y_axis_ant = ant_avg.iloc[:, 2]  # Mean return
y_err_ant = ant_std.iloc[:, 2]   # Standard deviation

x_axis_humanoid = humanoid_avg.iloc[:, 1]
y_axis_humanoid = humanoid_avg.iloc[:, 2]
y_err_humanoid = humanoid_std.iloc[:, 2]

# Placeholder values for expert and behavioral cloning performance (to be replaced with actual values if available)
expert_performance_ant = 4713.65
bc_performance_ant = 4683.13

expert_performance_humanoid = 10344.52
bc_performance_humanoid = 266.53

# Plot results
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Ant-v2 plot
axes[0].errorbar(x_axis_ant, y_axis_ant, yerr=y_err_ant, label="DAgger Policy", fmt='-o', capsize=3, color='b', ecolor='darkgray', elinewidth=2)
axes[0].axhline(y=expert_performance_ant, color='g', linestyle='--', label="Expert Policy")
axes[0].axhline(y=bc_performance_ant, color='r', linestyle='--', label="Behavioral Cloning")
axes[0].set_title("Ant-v2")
axes[0].set_xlabel("DAgger Iterations")
axes[0].set_ylabel("Mean Return")
axes[0].legend()
axes[0].grid(True)

# Humanoid-v2 plot
axes[1].errorbar(x_axis_humanoid, y_axis_humanoid, yerr=y_err_humanoid, label="DAgger Policy", fmt='-o', capsize=3, color='b', ecolor='lightgray', elinewidth=2)
axes[1].axhline(y=expert_performance_humanoid, color='g', linestyle='--', label="Expert Policy") 
axes[1].axhline(y=bc_performance_humanoid, color='r', linestyle='--', label="Behavioral Cloning")
axes[1].set_title("Humanoid-v2")
axes[1].set_xlabel("DAgger Iterations")
axes[1].set_ylabel("Mean Return")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()
