from MemoryBuffer import MemoryBuffer
from networks.DuelingNetwork import DuelingNetwork
from agents.DQNAgent import DQNAgent
from utils.AgentTrainingUtil import trainRLAgent

import gym
import torch
import matplotlib.pyplot as plt
import uuid

# Constants
DEVICE = torch.device('cpu')

BUFFER_CAPACITY = 10000
BATCH_SIZE = 64

MAX_EPSILON = 1
MIN_EPSILON = 0.001
EPSILON_DECAY = 0.999

GAMMA = 0.95

LEARNING_RATE = 0.0001

EPISODE_NUM = 200

env = gym.make('CartPole-v1')

if __name__ == '__main__':
    
    memory = MemoryBuffer(BUFFER_CAPACITY)
    network = DuelingNetwork(env.observation_space.shape, env.action_space.n, LEARNING_RATE)

    epsilon_info = (MAX_EPSILON, MIN_EPSILON, EPSILON_DECAY)
    agent = DQNAgent(network, memory, epsilon_info, GAMMA, env)
    
    data = trainRLAgent(agent, EPISODE_NUM, BATCH_SIZE, env)
    
    # Plotting on graph and saving to file
    plt.plot(*data)
    plt.title('Progress of DQN Agent')
    
    plt.xlabel('Episode Number')
    plt.ylabel('Reward')
    # -{EPISODE_NUM}-ep-{str(uuid.uuid4().hex[:4])}
    plt.savefig(f"figures/dueling-dqn--{EPISODE_NUM}-ep-{str(uuid.uuid4().hex[:4])}")
    
    plt.show()