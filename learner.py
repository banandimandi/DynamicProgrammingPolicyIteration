import numpy as np
class DPLearner():
    def __init__(self, env):
        self.env = env
        self.theta = 1e-6
        self.grid_size = env.size
        self.V= np.zeros((self.grid_size, self.grid_size)) # Initializes value function, values are arbitrary
        self.policy = {}
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                state = (i, j)
                self.policy[state] = {
                    a : .25 for a in range(self.env.action_space.n)
                }
    def value_iteration(self):
        while True:
            delta = 0
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    state = (i,j)
                    if self.env.is_terminal(state):
                        continue
                    v = self.V[state]

                    action_values = []
                    for action in range(self.env.action_space.n):
                        next_state, reward = self.env.transition(state, action)
                        action_values.append(
                            reward + self.env.gamma * self.V[next_state]
                        )
                    self.V[state] = max(action_values)
                    delta = max(delta, abs(v - self.V[state]))
            if delta < self.theta:
                break



        for i in range(self.grid_size):
            for j in range(self.grid_size):
                state = (i, j)
                if self.env.is_terminal(state):
                    self.policy[state] = {a: 0 for a in range(self.env.action_space.n)}
                    continue
                action_values = {}

                for action in range(self.env.action_space.n):
                    next_state, reward = self.env.transition(state, action)
                    action_values[action] = reward + self.env.gamma * self.V[next_state]
                best_action = max(action_values, key=action_values.get)
                self.policy[state] = {a: 1 if a == best_action else 0 for a in range(self.env.action_space.n)}
        return self.V, self.policy










