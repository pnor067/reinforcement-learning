import torch
import random

class DQNAgent(object):
    
    def __init__(self, network : torch.nn.Module, memory, epsilon_info, gamma, env):
        self.memory = memory
        self.network = network

        self.epsilon, self.min_epsilon, self.epsilon_decay = epsilon_info

        self.gamma = gamma
        self.env = env
    
    def chooseAction(self, state):
        
        # With probability Epsilon, select a random action
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        
        # Generate Expected Future Return
        state_tensor = torch.tensor(state)
        q_values = self.network.forward(state_tensor)
        
        # Select the action with the best estimated future return
        return torch.argmax(q_values).item()

    def learnFromMemory(self, batch_size):

        # Only begin learning when there is enough experience in buffer to sample from
        if len(self.memory.buffer) < batch_size:
            return
        
        # Sample batch of size BATCH_SIZE from past experiences
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        
        # Generate Q Values given state at time t and t + 1 
        q_values = self.network.forward(states)
        next_q_values = self.network.forward(next_states)
        
        # Get the q values using current model of the actual actions taken historically
        best_q_values = q_values[torch.arange(q_values.size(0)), actions]
        '''
        Intuitively what is trying to be achieved is 
            current_q_values[:, actions]
        Where we are trying to get the q values of the actions that were historically taken by the agent
        
        The line above using the method described in the link below achieves this
        https://stackoverflow.com/questions/53986301/how-can-i-select-single-indices-over-a-dimension-in-pytorch
        
        '''
        
        # For q values at time t + 1, return all the best actions in each state
        best_next_q_values = torch.max(next_q_values, dim=1).values
        
        # Compute the target q values based on bellman's equations
        expected_q_values = rewards + self.gamma * best_next_q_values * ~dones
        
        # Update the Network
        loss = self.network.loss(best_q_values, expected_q_values)
        self.network.optimizer.zero_grad()
        loss.backward()
        self.network.optimizer.step()
        
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.min_epsilon)