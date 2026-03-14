import numpy as np


class DPLearner():
    def __init__(self, env):
        self.env = env
        self.theta = 1e-6
        self.grid_size = env.size
        self.V = np.zeros((self.grid_size, self.grid_size))  # Initializes value function, values are arbitrary
        self.policy = {}
        self.policy_stable = []

        """ assigning uniform probability distribution """
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                state = (i, j)
                self.policy_stable.append(False)
                self.policy[state] = {
                    a: .25 for a in range(self.env.action_space.n)
                }

    def _policy_evaluation(self):

        while True:
            delta = 0
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    state = (i, j)
                    if self.env.is_terminal(state):
                        continue

                    v = self.V[state]

                    new_v = 0

                    for action, prob in self.policy[state].items():
                        next_state, reward = self.env.transition(state, action)
                        new_v += prob * (reward + self.env.gamma * self.V[next_state])
                    self.V[state] = new_v
                    delta = max(delta, abs(v - self.V[state]))
            if delta < self.theta:
                break
        return self.V

    def _policy_improvement(self):
        self.policy_stable = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                state = (i, j)
                action_values = {}
                if self.env.is_terminal(state):
                    continue
                for action in range(self.env.action_space.n):
                    next_state, reward = self.env.transition(state, action)
                    action_values[action] = reward + self.env.gamma * self.V[next_state]
                best_action = max(action_values, key=action_values.get)
                if best_action != max(self.policy[state], key=self.policy[state].get):
                    self.policy[state] = {a: 1 if a == best_action else 0 for a in range(self.env.action_space.n)}
                    self.policy_stable.append(False)
                else:
                    self.policy_stable.append(True)
                    continue
        return self.policy

    def policy_iteration(self):
        while True:
            self._policy_evaluation()
            self._policy_improvement()
            if all(self.policy_stable):
                break
        return self.V, self.policy
