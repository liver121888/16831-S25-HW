import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# File paths
data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data')

q1_dqn1 = os.path.join(data_path, "q1_dqn_1_LunarLander-v3_15-03-2025_17-41-19.csv")
q1_dqn2 = os.path.join(data_path, "q1_dqn_2_LunarLander-v3_15-03-2025_17-59-27.csv")
q1_dqn3 = os.path.join(data_path, "q1_dqn_3_LunarLander-v3_15-03-2025_18-10-03.csv")

q1_ddqn1 = os.path.join(data_path, "q1_doubledqn_1_LunarLander-v3_15-03-2025_18-30-13.csv")
q1_ddqn2 = os.path.join(data_path, "q1_doubledqn_2_LunarLander-v3_15-03-2025_18-33-21.csv")
q1_ddqn3 = os.path.join(data_path, "q1_doubledqn_3_LunarLander-v3_15-03-2025_18-47-06.csv")

# Function to read CSV and extract Step and Value columns
def read_data(file_path):
    df = pd.read_csv(file_path)
    # Convert Series to numpy arrays
    return df['Step'].to_numpy(), df['Value'].to_numpy()

# Q1: dqn vs ddqn

# Read data
step_q1_dqn1, value_q1_dqn1 = read_data(q1_dqn1)
step_q1_dqn2, value_q1_dqn2 = read_data(q1_dqn2)
step_q1_dqn3, value_q1_dqn3 = read_data(q1_dqn3)

step_q1_ddqn1, value_q1_ddqn1 = read_data(q1_ddqn1)
step_q1_ddqn2, value_q1_ddqn2 = read_data(q1_ddqn2)
step_q1_ddqn3, value_q1_ddqn3 = read_data(q1_ddqn3)

dqn_values = np.stack([value_q1_dqn1, value_q1_dqn2, value_q1_dqn3])
ddqn_values = np.stack([value_q1_ddqn1, value_q1_ddqn2, value_q1_ddqn3])

# print(dqn_values.shape)
# (3, 30)

dqn_mean = np.mean(dqn_values, axis=0)
dqn_std = np.std(dqn_values, axis=0)

ddqn_mean = np.mean(ddqn_values, axis=0)
ddqn_std = np.std(ddqn_values, axis=0)

# Print shapes
# print(step_q1_dqn1.shape, value_q1_dqn1.shape)
# print(dqn_mean.shape, dqn_std.shape)

# Plotting
plt.figure(figsize=(10, 6))

# Plot DQN with error bars
# plt.plot(step_q1_dqn1, dqn_mean, label='DQN', color='blue')
# plt.fill_between(step_q1_dqn1, dqn_mean - dqn_std, dqn_mean + dqn_std, color='blue', alpha=0.2)

# Plot DDQN with error bars
# plt.plot(step_q1_ddqn1, ddqn_mean, label='DDQN', color='orange')
# plt.fill_between(step_q1_ddqn1, ddqn_mean - ddqn_std, ddqn_mean + ddqn_std, color='orange', alpha=0.2)

# Plot DQN with error bars
plt.errorbar(step_q1_dqn1, dqn_mean, yerr=dqn_std, label='DQN', color='blue', fmt='-o', markersize=5, capsize=5)

# Plot DDQN with error bars
plt.errorbar(step_q1_ddqn1, ddqn_mean, yerr=ddqn_std, label='DDQN', color='orange', fmt='-o', markersize=5, capsize=5)

# Labels and Title
plt.xlabel('Time Steps (scientific notation)')
plt.ylabel('Average Return')
plt.title('Learning Curve: DQN vs DDQN on LunarLander-v3')
plt.legend()
plt.grid(True)

# Use scientific notation for the x-axis
plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
plt.show()

# Q2: actor-critic

q2_ac = os.path.join(data_path, "q2_10_10_CartPole-v0_17-03-2025_16-13-14.csv")

step_q2_ac, value_q2_ac = read_data(q2_ac)
plt.figure(figsize=(10, 6))

plt.plot(step_q2_ac, value_q2_ac, label='Actor-Critic', color='orange')
plt.xlabel('Time Steps (scientific notation)')
plt.ylabel('Evaluation Return')
plt.title('Learning Curve: Actor-Critic on  Cartpole-v0')
plt.legend()
plt.grid(True)

plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
plt.show()

# Q3: actor-critic

q3_ac = os.path.join(data_path, "q3_10_10_InvertedPendulum-v4_17-03-2025_16-15-09.csv")
step_q3_ac, value_q3_ac = read_data(q3_ac)
plt.figure(figsize=(10, 6))
plt.plot(step_q3_ac, value_q3_ac, label='Actor-Critic', color='blue')
plt.xlabel('Time Steps (scientific notation)')
plt.ylabel('Evaluation Return')
plt.title('Learning Curve: Actor-Critic on InvertedPendulum-v4')
plt.legend()
plt.grid(True)

plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
plt.show()
