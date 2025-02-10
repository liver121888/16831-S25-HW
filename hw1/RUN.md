# Commands to produce the result

## 1.2

### Ant
python rob831/scripts/run_hw1.py \
--expert_policy_file rob831/policies/experts/Ant.pkl \
--env_name Ant-v2 --exp_name bc_ant --n_iter 1 \
--expert_data rob831/expert_data/expert_data_Ant-v2.pkl \
--video_log_freq -1

### Humanoid
python rob831/scripts/run_hw1.py \
--expert_policy_file rob831/policies/experts/Humanoid.pkl \
--env_name Humanoid-v2 --exp_name bc_humanoid --n_iter 1 \
--expert_data rob831/expert_data/expert_data_Humanoid-v2.pkl \
--video_log_freq -1

### Walker2d-v2
python rob831/scripts/run_hw1.py \
--expert_policy_file rob831/policies/experts/Walker2d.pkl \
--env_name Walker2d-v2 --exp_name bc_walker2d --n_iter 1 \
--expert_data rob831/expert_data/expert_data_Walker2d-v2.pkl \
--video_log_freq -1

### Hopper-v2
python rob831/scripts/run_hw1.py \
--expert_policy_file rob831/policies/experts/Hopper.pkl \
--env_name Hopper-v2 --exp_name bc_hopper --n_iter 1 \
--expert_data rob831/expert_data/expert_data_Hopper-v2.pkl \
--video_log_freq -1

### HalfCheetah-v2
python rob831/scripts/run_hw1.py \
--expert_policy_file rob831/policies/experts/HalfCheetah.pkl \
--env_name HalfCheetah-v2 --exp_name bc_halfCheetah --n_iter 1 \
--expert_data rob831/expert_data/expert_data_HalfCheetah-v2.pkl \
--video_log_freq -1

## 1.3

### Ant
python rob831/scripts/run_hw1.py \
--expert_policy_file rob831/policies/experts/Ant.pkl \
--env_name Ant-v2 --exp_name bc_ant --n_iter 1 \
--expert_data rob831/expert_data/expert_data_Ant-v2.pkl \
--video_log_freq -1 \
--eval_batch_size 5000

### Humanoid
python rob831/scripts/run_hw1.py \
--expert_policy_file rob831/policies/experts/Humanoid.pkl \
--env_name Humanoid-v2 --exp_name bc_humanoid --n_iter 1 \
--expert_data rob831/expert_data/expert_data_Humanoid-v2.pkl \
--video_log_freq -1 \
--eval_batch_size 5000

## 1.4

### Original
python rob831/scripts/run_hw1.py \
--expert_policy_file rob831/policies/experts/Humanoid.pkl \
--env_name Humanoid-v2 --exp_name bc_humanoid --n_iter 1 \
--expert_data rob831/expert_data/expert_data_Humanoid-v2.pkl \
--video_log_freq -1 \
--ep_len 1000 \
--eval_batch_size 5000

### My params
python rob831/scripts/run_hw1.py \
--expert_policy_file rob831/policies/experts/Humanoid.pkl \
--env_name Humanoid-v2 --exp_name bc_humanoid --n_iter 1 \
--expert_data rob831/expert_data/expert_data_Humanoid-v2.pkl \
--video_log_freq -1 \
--ep_len 1000 \
--train_batch_size 500 \
--eval_batch_size 5000 \
--n_layers 5 \
--size 128

## 2.2

### Ant
python rob831/scripts/run_hw1.py \
--expert_policy_file rob831/policies/experts/Ant.pkl \
--env_name Ant-v2 --exp_name dagger_ant --n_iter 10 \
--do_dagger --expert_data rob831/expert_data/expert_data_Ant-v2.pkl \
--ep_len 1000 \
--eval_batch_size 5000 \
--video_log_freq -1

### Humanoid
python rob831/scripts/run_hw1.py \
--expert_policy_file rob831/policies/experts/Humanoid.pkl \
--env_name Humanoid-v2 --exp_name dagger_humanoid --n_iter 100 \
--do_dagger --expert_data rob831/expert_data/expert_data_Humanoid-v2.pkl \
--video_log_freq -1 \
--ep_len 1000 \
--train_batch_size 1000 \
--eval_batch_size 5000 \
--n_layers 5 \
--size 128

