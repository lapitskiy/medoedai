from utils.env import dqncfg

from tensorflow.keras.models import load_model

import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from collections import deque
import gym
from gym import spaces
from gym.utils import seeding


class DQNSolver:
    def __init__(self, observation_space, action_space, load=False, model_path="dqn_model.h5"):
        self.exploration_rate = EXPLORATION_MAX
        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        if load:
            self.model = load_model(model_path)
            print("Модель загружена из", model_path)
        else:
            self.model = Sequential()
            self.model.add(Dense(24, input_shape=(observation_space,), activation="relu"))
            self.model.add(Dense(24, activation="relu"))
            self.model.add(Dense(action_space, activation="linear"))
            self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))
            print("Создана новая модель")

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)


if __name__ == "__main__":

    env = CryptoTradingEnv(df)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    load_previous_model = False  # Установите True, чтобы загрузить существующую модель
    dqn_solver = DQNSolver(observation_space, action_space, load=load_previous_model)

    episodes = 1000
    for episode in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        while True:
            action = dqn_solver.act(state)
            state_next, reward, terminal, info = env.step(action)
            state_next = np.reshape(state_next, [1, observation_space])
            dqn_solver.remember(state, action, reward, state_next, terminal)
            state = state_next
            if terminal:
                print("Эпизод", episode, "завершен")
                break
            dqn_solver.experience_replay()

        # Сохраняем модель после каждых 10 эпизодов
        if episode % 10 == 0:
            dqn_solver.model.save("dqn_model.h5")
            print("Модель сохранена после эпизода", episode)