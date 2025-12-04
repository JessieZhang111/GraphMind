# RL Trainer

class RLTrainer:
    def __init__(self, environment):
        self.environment = environment
        self.done = False
        self.state = self.environment.reset()

    def training_loop(self, num_episodes):
        for episode in range(num_episodes):
            self.state = self.environment.reset()
            total_reward = 0
            while not self.done:
                action = self.select_action(self.state)
                self.state, reward, self.done, _ = self.environment.step(action)
                total_reward += reward
            print(f'Episode {episode + 1}: Total Reward = {total_reward}')

    def select_action(self, state):
        # Implement action selection logic here (e.g., ε-greedy)
        pass
