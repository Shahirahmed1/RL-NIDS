from utils.environment import NetworkEnv
from dqn_agent import DQNAgent
import matplotlib.pyplot as plt
import numpy as np
print("Starting DQN Training... Please wait...")
env = NetworkEnv()
agent = DQNAgent(state_size=3, action_size=2)

episodes = 500  # number of episodes can change it to 300, 100, or 50
# depending on the training time you want to spend
rewards = []

for ep in range(episodes):
    state, _ = env.reset()
    total_reward = 0

    for step in range(10):  # short episodes can be reduce to 5, 3, or 1
        # depending on the training time you want to spend
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.learn(state, action, reward, next_state)
        total_reward += reward
        state = next_state

    rewards.append(total_reward)

print("DQN Training Completed!")

# Plotting Rewards
plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('DQN Training Progress')
plt.grid()
plt.savefig('results/dqn_rewards.png')
plt.show()
