from MemoryBuffer import MemoryBuffer
from networks.Network import Network
from agents.DoubleDQNAgent import DoubleDQNAgent
from utils.AgentTrainingUtil import trainRLAgent

import gym
import matplotlib.pyplot as plt
import uuid

# Constants
BUFFER_CAPACITY = 10000
BATCH_SIZE = 64

MAX_EPSILON = 1
MIN_EPSILON = 0.001
EPSILON_DECAY = 0.999

GAMMA = 0.95
TAU = 0.01

LEARNING_RATE = 0.0001

EPISODE_NUM = 50

env = gym.make('CartPole-v1')

def main():

    memory = MemoryBuffer(BUFFER_CAPACITY)
    main_network = Network(env.observation_space.shape, env.action_space.n, LEARNING_RATE)
    target_network = Network(env.observation_space.shape, env.action_space.n, LEARNING_RATE)

    epsilon_info = (MAX_EPSILON, MIN_EPSILON, EPSILON_DECAY)
    
    agent = DoubleDQNAgent(main_network, target_network, memory, epsilon_info, GAMMA, TAU, env)
    
    data = trainRLAgent(agent, EPISODE_NUM, BATCH_SIZE, env)
    
    # Plotting on graph and saving to file
    plt.plot(*data)
    plt.title('Progress of DQN Agent')
    
    plt.xlabel('Episode Number')
    plt.ylabel('Reward')
    
    plt.savefig(f"figures/double-dqn-{EPISODE_NUM}-ep-{str(uuid.uuid4().hex[:4])}")
    
    plt.show()

if __name__ == '__main__':
    main()