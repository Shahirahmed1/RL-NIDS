from utils.environment import NetworkEnv
from q_learning_agent import QLearningAgent
import numpy as np
import matplotlib.pyplot as plt

env = NetworkEnv()
agent = QLearningAgent(state_size=2, action_size=2)

episodes = 1000
rewards = []

for ep in range(episodes):
    state = np.random.choice([0, 1])
    total_reward = 0

    for step in range(50):
        action = agent.choose_action(state)
        next_state = np.random.choice([0, 1])
        reward = 1 if (state == 1 and action == 1) or (state == 0 and action == 0) else -1
        agent.learn(state, action, reward, next_state)
        total_reward += reward
        state = next_state

    rewards.append(total_reward)

print("Training completed!")

# Plotting rewards
plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Training Progress')
plt.grid()
plt.savefig('results/q_learning_rewards.png')  # Save to file
plt.show()
