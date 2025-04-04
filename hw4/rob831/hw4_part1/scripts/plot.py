import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Function to read CSV and extract Step and Value columns
def read_data(file_path):
    df = pd.read_csv(file_path)
    # Convert Series to numpy arrays
    return df['Step'].to_numpy(), df['Value'].to_numpy()


# File paths
data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data')

# Q3
q3_1= os.path.join(data_path, "hw4_q3_obstacles_obstacles-hw4_part1-v0_02-04-2025_15-09-01.csv")
q3_2 = os.path.join(data_path, "hw4_q3_reacher_reacher-hw4_part1-v0_02-04-2025_15-12-23.csv")
q3_3 = os.path.join(data_path, "hw4_q3_cheetah_cheetah-hw4_part1-v0_02-04-2025_15-31-31.csv")

# Read data
step_q3_1, value_q3_1 = read_data(q3_1)
step_q3_2, value_q3_2 = read_data(q3_2)
step_q3_3, value_q3_3 = read_data(q3_3)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(step_q3_1, value_q3_1, label='obstacles', color='orange')
plt.xlabel('Iteration')
plt.ylabel('Evaluation Return')
plt.title('Learning Curve: Eval_AverageReturn')
plt.legend()
plt.grid(True)
plt.show()

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(step_q3_2, value_q3_2, label='reacher', color='orange')
plt.xlabel('Iteration')
plt.ylabel('Evaluation Return')
plt.title('Learning Curve: Eval_AverageReturn')
plt.legend()
plt.grid(True)
plt.show()

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(step_q3_3, value_q3_3, label='cheetah', color='orange')
plt.xlabel('Iteration')
plt.ylabel('Evaluation Return')
plt.title('Learning Curve: Eval_AverageReturn')
plt.legend()
plt.grid(True)
plt.show()


# Q4

q4_1 = os.path.join(data_path, "hw4_q4_reacher_horizon5_reacher-hw4_part1-v0_02-04-2025_19-11-48.csv")
q4_2 = os.path.join(data_path, "hw4_q4_reacher_horizon15_reacher-hw4_part1-v0_02-04-2025_19-15-22.csv")
q4_3 = os.path.join(data_path, "hw4_q4_reacher_horizon30_reacher-hw4_part1-v0_03-04-2025_23-07-52.csv")
q4_4 = os.path.join(data_path, "hw4_q4_reacher_numseq100_reacher-hw4_part1-v0_02-04-2025_19-46-33.csv")
q4_5 = os.path.join(data_path, "hw4_q4_reacher_numseq1000_reacher-hw4_part1-v0_02-04-2025_19-51-23.csv")
q4_6 = os.path.join(data_path, "hw4_q4_reacher_ensemble1_reacher-hw4_part1-v0_02-04-2025_20-02-32.csv")
q4_7 = os.path.join(data_path, "hw4_q4_reacher_ensemble3_reacher-hw4_part1-v0_02-04-2025_20-06-40.csv")
q4_8 = os.path.join(data_path, "hw4_q4_reacher_ensemble5_reacher-hw4_part1-v0_02-04-2025_20-17-31.csv")

# Read data
step_q4_1, value_q4_1 = read_data(q4_1)
step_q4_2, value_q4_2 = read_data(q4_2)
step_q4_3, value_q4_3 = read_data(q4_3)
step_q4_4, value_q4_4 = read_data(q4_4)
step_q4_5, value_q4_5 = read_data(q4_5)
step_q4_6, value_q4_6 = read_data(q4_6)
step_q4_7, value_q4_7 = read_data(q4_7)
step_q4_8, value_q4_8 = read_data(q4_8)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(step_q4_1, value_q4_1, label='horizon=5', color='orange')
plt.plot(step_q4_2, value_q4_2, label='horizon=15', color='blue')
plt.plot(step_q4_3, value_q4_3, label='horizon=30', color='green')
plt.xlabel('Iteration')
plt.ylabel('Evaluation Return')
plt.title('Learning Curve: Eval_AverageReturn')
plt.legend()
plt.grid(True)
plt.show()

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(step_q4_4, value_q4_4, label='numseq=100', color='orange')
plt.plot(step_q4_5, value_q4_5, label='numseq=1000', color='blue')
plt.xlabel('Iteration')
plt.ylabel('Evaluation Return')
plt.title('Learning Curve: Eval_AverageReturn')
plt.legend()
plt.grid(True)
plt.show()

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(step_q4_6, value_q4_6, label='ensemble=1', color='orange')
plt.plot(step_q4_7, value_q4_7, label='ensemble=3', color='blue')
plt.plot(step_q4_8, value_q4_8, label='ensemble=5', color='green')
plt.xlabel('Iteration')
plt.ylabel('Evaluation Return')
plt.title('Learning Curve: Eval_AverageReturn')
plt.legend()
plt.grid(True)
plt.show()

# Q5
q5_1 = os.path.join(data_path, "hw4_q5_cheetah_random_cheetah-hw4_part1-v0_02-04-2025_20-35-23.csv")
q5_2 = os.path.join(data_path, "hw4_q5_cheetah_cem_2_cheetah-hw4_part1-v0_02-04-2025_17-57-03.csv")
q5_3 = os.path.join(data_path, "hw4_q5_cheetah_cem_4_cheetah-hw4_part1-v0_02-04-2025_18-21-24.csv")

# Read data
step_q5_1, value_q5_1 = read_data(q5_1)
step_q5_2, value_q5_2 = read_data(q5_2)
step_q5_3, value_q5_3 = read_data(q5_3)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(step_q5_1, value_q5_1, label='random', color='orange')
plt.plot(step_q5_2, value_q5_2, label='cem=2', color='blue')
plt.plot(step_q5_3, value_q5_3, label='cem=4', color='green')
plt.xlabel('Iteration')
plt.ylabel('Evaluation Return')
plt.title('Learning Curve: Eval_AverageReturn')
plt.legend()
plt.grid(True)
plt.show()
