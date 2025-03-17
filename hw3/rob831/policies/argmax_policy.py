import numpy as np
import torch

from rob831.infrastructure import pytorch_util as ptu 

class ArgMaxPolicy(object):

    def __init__(self, critic):
        self.critic = critic

    def get_action(self, obs):
        if len(obs.shape) > 3:
            observation = obs
        else:
            observation = obs[None]
        
        ## TODO return the action that maxinmizes the Q-value 
        # at the current observation as the output
        observation = ptu.from_numpy(observation)
        self.critic.q_net_target.eval()
        with torch.no_grad():
            action = self.critic.q_net_target(observation).argmax(dim=1)
        action = ptu.to_numpy(action)
        return action.squeeze()
