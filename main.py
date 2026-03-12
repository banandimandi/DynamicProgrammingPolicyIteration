from grid_world import GridWorldEnv
from learner import DPLearner
import gymnasium as gym
import numpy as np
import pygame

def main():
    env = GridWorldEnv(render_mode="human", size=5)
    env.reset() ## target_location only initialized in reset. is_terminal() fails if reset() is not called here
    learner = DPLearner(env)

    V, policy = learner.value_iteration()

    for episode in range(1, 10):
        state, info = env.reset()
        done = False

        while not done:
            state = tuple(state)
            action = max(policy[state], key = policy[state].get)

            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            state = observation

if __name__ == "__main__":
    main()
