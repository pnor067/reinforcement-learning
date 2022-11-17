from collections import deque, namedtuple
import random
import torch
import numpy as np


class MemoryBuffer(object):

    def __init__(self, max_capacity):
        self.buffer = deque([],maxlen=max_capacity)
    
    def add(self, *experience):
        self.buffer.append(experience)
        
    def sample(self, batch_size):
        
        # Randomly sample experiences from buffer of size batch_size
        experienceBatch = random.sample(self.buffer, batch_size)

        # Destructure batch experiences into tuples of _
        # eg. tuples of states, tuples of actions...
        states, actions, rewards, next_states, dones = zip(*experienceBatch)
        
        # Convert from _ tuples to _ tensors
        # eg. states tuple to states tensor
        states = torch.tensor(np.asarray(states), dtype=torch.float32) # shape: 64 [batch_size] x 4 [observations]
        actions = torch.tensor(actions, dtype=torch.long) # shape: 64 [batch_size] x 1 [action_taken]
        rewards = torch.tensor(rewards, dtype=torch.float32) # shape: 64 [batch_size] x 1 [reward_received]
        next_states = torch.tensor(np.asarray(next_states), dtype=torch.float32) # shape: 64 [batch_size] x 4 [new_observations]
        dones = torch.tensor(dones) # shape: 64 [batch_size] x 1 [is_done]
        
        return (states, actions, rewards, next_states, dones)