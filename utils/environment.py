from traffic_simulator import generate_traffic

class NetworkEnv:
    def __init__(self):
        pass

    def reset(self):
        self.state, self.label = generate_traffic()
        return self.state, self.label  # ✅ Correct: Return both state and label


    def step(self, action):
        reward = 1 if action == self.label else -1
        next_state, next_label = generate_traffic()
        done = False  # You can set done=True after certain steps if you want
        return next_state, reward, done
