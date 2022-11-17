

def trainRLAgent(agent, episode_num, batch_size, env):
    
    # Track the reward over EPISODE_NUM episodes
    historical_reward = []
    
    for episode in range(0, episode_num):
        
        # Initial State
        state, _ = env.reset()

        episode_reward = 0
        
        while True:
            
            action = agent.chooseAction(state)

            # Take the next action and observe the effect
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # Add the experience to the memory buffer
            agent.memory.add(state, action, reward, next_state, terminated)
            
            # Train the neural network when the size of the memory buffer is greater than or equal to the batch size
            for _ in range(1, 10):
                agent.learnFromMemory(batch_size)

            state = next_state
            episode_reward += reward
            
            if(terminated or truncated):
                break
        
        historical_reward.append(episode_reward)
        
        print(f"Episode #{episode} Reward {episode_reward}")

    # Data collected during run, for plotting
    episode_data = range(0, episode_num)
    reward_data = historical_reward

    return (episode_data, reward_data)