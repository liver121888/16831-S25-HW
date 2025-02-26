import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# File paths
data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data')

sb_no_rtg_dsa = os.path.join(data_path, "q1_sb_no_rtg_dsa_CartPole-v0_24-02-2025_21-43-37.csv")
sb_rtg_dsa = os.path.join(data_path, "q1_sb_rtg_dsa_CartPole-v0_24-02-2025_21-46-45.csv")
sb_rtg_na = os.path.join(data_path, "q1_sb_rtg_na_CartPole-v0_24-02-2025_21-48-37.csv")

# Function to read CSV and extract Step and Value columns
def read_data(file_path):
    df = pd.read_csv(file_path)
    # Convert Series to numpy arrays
    return df['Step'].to_numpy(), df['Value'].to_numpy()

# Q1: Small batch

# Read data
step_no_rtg_dsa, value_no_rtg_dsa = read_data(sb_no_rtg_dsa)
step_rtg_dsa, value_rtg_dsa = read_data(sb_rtg_dsa)
step_rtg_na, value_rtg_na = read_data(sb_rtg_na)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(step_no_rtg_dsa, value_no_rtg_dsa, label='No RTG DSA', color='blue')
plt.plot(step_rtg_dsa, value_rtg_dsa, label='RTG DSA', color='green')
plt.plot(step_rtg_na, value_rtg_na, label='RTG NA', color='red')

# Labels and Title
plt.xlabel('Step')
plt.ylabel('Value')
plt.title('Small Batch Experiments on CartPole-v0')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Q1: Large batch

lb_no_rtg_dsa = os.path.join(data_path, "q1_lb_no_rtg_dsa_CartPole-v0_24-02-2025_21-50-04.csv")
lb_rtg_dsa = os.path.join(data_path, "q1_lb_rtg_dsa_CartPole-v0_24-02-2025_21-54-07.csv")
lb_rtg_na = os.path.join(data_path, "q1_lb_rtg_na_CartPole-v0_24-02-2025_22-10-26.csv")

# Read data
step_no_rtg_dsa, value_no_rtg_dsa = read_data(lb_no_rtg_dsa)
step_rtg_dsa, value_rtg_dsa = read_data(lb_rtg_dsa)
step_rtg_na, value_rtg_na = read_data(lb_rtg_na)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(step_no_rtg_dsa, value_no_rtg_dsa, label='No RTG DSA', color='blue')
plt.plot(step_rtg_dsa, value_rtg_dsa, label='RTG DSA', color='green')
plt.plot(step_rtg_na, value_rtg_na, label='RTG NA', color='red')

# Labels and Title
plt.xlabel('Step')
plt.ylabel('Value')
plt.title('Large Batch Experiments on CartPole-v0')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Q2: InvertedPendulum
data = os.path.join(data_path, "q2_b1000_r5e-3_InvertedPendulum-v4_24-02-2025_22-46-08.csv")
step, value = read_data(data)
# Plotting
plt.figure(figsize=(10, 6))
plt.plot(step, value, label='b1000_r5e-3', color='blue')

# Labels and Title
plt.xlabel('Step')
plt.ylabel('Value')
plt.title('Result on InvertedPendulum-v4')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Q3: LunarLander
lunar = os.path.join(data_path, "q3_b10000_r0.005_LunarLanderContinuous-v2_25-02-2025_00-15-42.csv")
step, value = read_data(lunar)
# Plotting
plt.figure(figsize=(10, 6))
plt.plot(step, value, label='b40000_r0.005', color='blue')

# Labels and Title
plt.xlabel('Step')
plt.ylabel('Value')
plt.title('Result on LunarLanderContinuous-v2')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Q4: HalfCheetah

b10000_lr0005 = os.path.join(data_path, "q4_search_b10000_lr0.005_rtg_nnbaseline_HalfCheetah-v4_24-02-2025_23-05-42.csv")
b10000_lr001 = os.path.join(data_path, "q4_search_b10000_lr0.01_rtg_nnbaseline_HalfCheetah-v4_25-02-2025_11-53-37.csv")
b10000_lr002 = os.path.join(data_path, "q4_search_b10000_lr0.02_rtg_nnbaseline_HalfCheetah-v4_25-02-2025_12-04-36.csv")

b30000_lr0005 = os.path.join(data_path, "q4_search_b30000_lr0.005_rtg_nnbaseline_HalfCheetah-v4_24-02-2025_23-09-09.csv")
b30000_lr001 = os.path.join(data_path, "q4_search_b30000_lr0.01_rtg_nnbaseline_HalfCheetah-v4_24-02-2025_23-54-47.csv")
b30000_lr002 = os.path.join(data_path, "q4_search_b30000_lr0.02_rtg_nnbaseline_HalfCheetah-v4_25-02-2025_12-15-24.csv")

b50000_lr0005 = os.path.join(data_path, "q4_search_b50000_lr0.005_rtg_nnbaseline_HalfCheetah-v4_25-02-2025_12-44-41.csv")
b50000_lr001 = os.path.join(data_path, "q4_search_b50000_lr0.01_rtg_nnbaseline_HalfCheetah-v4_25-02-2025_13-31-36.csv")
b50000_lr002 = os.path.join(data_path, "q4_search_b50000_lr0.02_rtg_nnbaseline_HalfCheetah-v4_25-02-2025_14-19-36.csv")

# Read data
step_0, value_0 = read_data(b10000_lr0005)
step_1, value_1 = read_data(b10000_lr001)
step_2, value_2 = read_data(b10000_lr002)
step_3, value_3 = read_data(b30000_lr0005)
step_4, value_4 = read_data(b30000_lr001)
step_5, value_5 = read_data(b30000_lr002)
step_6, value_6 = read_data(b50000_lr0005)
step_7, value_7 = read_data(b50000_lr001)
step_8, value_8 = read_data(b50000_lr002)


# Plotting
plt.figure(figsize=(10, 6))
plt.plot(step_0, value_0, label='b10000_lr0.005', color='blue')
plt.plot(step_1, value_1, label='b10000_lr0.01', color='green')
plt.plot(step_2, value_2, label='b10000_lr0.02', color='red')
plt.plot(step_3, value_3, label='b30000_lr0.005', color='magenta')
plt.plot(step_4, value_4, label='b30000_lr0.01', color='cyan')
plt.plot(step_5, value_5, label='b30000_lr0.02', color='gold')
plt.plot(step_6, value_6, label='b50000_lr0.005', color='black')
plt.plot(step_7, value_7, label='b50000_lr0.01', color='purple')
plt.plot(step_8, value_8, label='b50000_lr0.02', color='orange')

# Labels and Title
plt.xlabel('Step')
plt.ylabel('Value')
plt.title('Search through batch size and learning rate on HalfCheetah-v4')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

exp_0 = os.path.join(data_path, "q4_b30000_r0.02_HalfCheetah-v4_25-02-2025_15-21-05.csv")
exp_rtg = os.path.join(data_path, "q4_b30000_r0.02_rtg_HalfCheetah-v4_25-02-2025_15-46-23.csv")
exp_nnbaseline = os.path.join(data_path, "q4_b30000_r0.02_nnbaseline_HalfCheetah-v4_25-02-2025_16-15-30.csv")
exp_rtg_nnbaseline = os.path.join(data_path, "q4_b30000_r0.02_rtg_nnbaseline_HalfCheetah-v4_25-02-2025_16-40-00.csv")

step_0, value_0 = read_data(exp_0)
step_1, value_1 = read_data(exp_rtg)
step_2, value_2 = read_data(exp_nnbaseline)
step_3, value_4 = read_data(exp_rtg_nnbaseline)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(step_0, value_0, label='exp', color='blue')
plt.plot(step_1, value_1, label='exp_rtg', color='green')
plt.plot(step_2, value_2, label='exp_nnbaseline', color='red')
plt.plot(step_3, value_4, label='exp_rtg_nnbaseline', color='gold')

# Labels and Title
plt.xlabel('Step')
plt.ylabel('Value')
plt.title('Various experiments on HalfCheetah-v4')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Q5: Hopper
lambda_0 = os.path.join(data_path, "q5_b2000_r0.001_lambda0_Hopper-v4_24-02-2025_23-21-20.csv")
lambda_095 = os.path.join(data_path, "q5_b2000_r0.001_lambda0.95_Hopper-v4_24-02-2025_23-36-31.csv")
lambda_099 = os.path.join(data_path, "q5_b2000_r0.001_lambda0.99_Hopper-v4_24-02-2025_23-50-24.csv")
lambda_1 = os.path.join(data_path, "q5_b2000_r0.001_lambda1.0_Hopper-v4_25-02-2025_00-01-46.csv")

# Read data
step_0, value_0 = read_data(lambda_0)
step_095, value_095 = read_data(lambda_095)
step_099, value_099 = read_data(lambda_099)
step_1, value_1 = read_data(lambda_1)


# Plotting
plt.figure(figsize=(10, 6))
plt.plot(step_0, value_0, label='lambda 0.0', color='blue')
plt.plot(step_095, value_095, label='lambda 0.95', color='green')
plt.plot(step_099, value_099, label='lambda 0.99', color='red')
plt.plot(step_1, value_1, label='lambda 1.0', color='magenta')

# Labels and Title
plt.xlabel('Step')
plt.ylabel('Value')
plt.title('Generalized advantage estimation experiments on Hopper-v4')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

