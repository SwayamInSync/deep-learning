import os
from mss import mss
from pyautogui import press, click
import cv2
import numpy as np
import pytesseract
import time
from gym import Env
from gym.spaces import Box, Discrete
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common import env_checker
from stable_baselines3 import A2C, DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack


class DinoEnv(Env):
    def __init__(self):
        super().__init__()
        self.observation_space = Box(low=0, high=255, shape=(1, 100, 100), dtype=np.uint8)
        self.action_space = Discrete(3) # jump, dodge, noop
        self.capture_screen = mss()
        self.game_area = {'top': 300, 'left': 0, 'width': 600, 'height': 500}
        self.game_over_area = {'top': 400, 'left': 400, 'width': 660, 'height': 80}
    def step(self, action):
        action_map = {
            0: 'space',
            1: 'down',
            2: 'noop'
        }

        if action != 2:
            press(action_map[action])

        done, done_cap = self.is_done()
        observation = self.get_observation()
        reward = 1
        info = {}
        return observation, reward, done, info

    def reset(self):
        time.sleep(1)
        click(x=150, y=150)
        press('space')
        return self.get_observation()

    def render(self, mode="human"):
        cv2.imshow('Game', self.current_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.close()
    def close(self):
        cv2.destroyAllWindows()
    def get_observation(self):
        raw = np.array(self.capture_screen.grab(self.game_area))[:, :, :3].astype(np.uint8)
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (100, 100))
        channel = np.reshape(resized, (1, 100, 100))
        return channel
    def is_done(self):
        done_cap = np.array(self.capture_screen.grab(self.game_over_area))
        done = False
        res = pytesseract.image_to_string(done_cap)[-5:]
        if res.strip() == 'OVER':
            done = True
        return done, done_cap

def try_env(env, model=None):
    for episode in range(10):
        obs = env.reset()
        done = False
        total_reward   = 0
        while not done:
            if model is None:
                obs, reward,  done, info =  env. step(env.action_space.sample())
            else:
                action, _ = model.predict(obs)
                obs, reward, done, info = env.step(int(action))
            total_reward  += reward
        print(f'Total Reward for episode {episode} is {total_reward}')

def verify_env(env):
    env_checker.check_env(env)

class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True

CHECKPOINT_DIR = './train/'
callback = TrainAndLoggingCallback(check_freq=1000, save_path=CHECKPOINT_DIR)

env = DinoEnv()
vec_env = DummyVecEnv([lambda : env])
model = A2C('CnnPolicy', vec_env, verbose=1, buffer_size=1200000, learning_starts=1000)

def train():
    model.learn(total_timesteps=20000, callback=callback)

def load(path, env):
    model = A2C.load(path, env=env)
    return model


time.sleep(2)
train()
# model2 = load('train/model_20000.zip', env)
#   try_env(env, model2)




