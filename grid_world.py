from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame


class Actions(Enum):
    right = 0
    up = 1
    left = 2
    down = 3


class GridWorldEnv(gym.Env):
    metadata = {"render_mode": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5, gamma=.95):
        self.size = size
        self.window_size = 512
        self.gamma = gamma

        self.observation_space = spaces.Box(0, size - 1, shape=(2,), dtype=int)

        self.action_space = spaces.Discrete(4)

        self._action_to_direction = {
            Actions.right.value: np.array([1, 0]),
            Actions.up.value: np.array([0, 1]),
            Actions.left.value: np.array([-1, 0]),
            Actions.down.value: np.array([0, -1])
        }

        assert render_mode is None or render_mode in self.metadata["render_mode"]

        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def _get_obs(self):
        return self._agent_location

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)
        self._agent_location = np.array([2, 0])

        self._target_location = np.array([4, 4])  #self._agent_location

        #while np.array_equal(self._agent_location, self._target_location):
        #    self._target_location = self.np_random.integers(
        #        0, self.size, size = 2, dtype = int
        #    )
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()
        return observation, info

    def step(self, action):
        direction = self._action_to_direction[action]

        """ NEEDS EXPLANATION"""
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = 1 if terminated else -1  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()
        return observation, reward, terminated, False, info

    def is_terminal(self, state):
        return np.array_equal(state, self._target_location)


    def transition(self, state, action):
        direction = self._action_to_direction[action]
        next_state = np.clip(
            np.array(state) + direction, 0, self.size - 1
        )
        next_state = tuple(next_state)
        reward = 1 if self.is_terminal(next_state) else - 1
        return next_state, reward

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
                self.window_size / self.size
        )

        """NEEDS EXPLANATION"""
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size)
            )
        )
        """NEEDS EXPLANATION"""
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=2,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=2,
            )
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
