from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense
import random
import numpy as np
import pandas as pd
import collections


class DQNAgent(object):
    def __init__(self, params):
        self.reward = 0
        self.gamma = 0.9
        self.dataframe = pd.DataFrame()
        self.short_memory = np.array([])
        self.agent_target = 1
        self.agent_predict = 0
        self.learning_rate = params['learning_rate']
        self.epsilon = 1
        self.actual = []
        self.first_layer = params['first_layer_size']
        self.second_layer = params['second_layer_size']
        self.third_layer = params['third_layer_size']
        self.memory = collections.deque(maxlen=params['memory_size'])
        self.weights = params['weights_path']
        self.load_weights = params['load_weights']
        self.model = self.network()

    def network(self):
        model = Sequential()
        model.add(Dense(output_dim=self.first_layer, activation='relu', input_dim=6))
        model.add(Dense(output_dim=self.second_layer, activation='relu'))
        model.add(Dense(output_dim=self.third_layer, activation='relu'))
        model.add(Dense(output_dim=3, activation='softmax'))
        opt = Adam(self.learning_rate)
        model.compile(loss='mse', optimizer=opt)

        if self.load_weights:
            model.load_weights(self.weights)
        return model

    def get_state(self, dino, cacti, pteras):  # TODO donner plus de vision au cerveau de l'IA
        ennemi_plus_proche = None
        distance_min = 1000
        for c in cacti:
            if distance_min > c.rect.left > 10:
                ennemi_plus_proche = c
        for p in pteras:
            if distance_min > p.rect.left > 10:
                ennemi_plus_proche = p
        state = [
            ennemi_plus_proche is not None and ennemi_plus_proche.rect.left <= 600 and ennemi_plus_proche.rect.bottom >= 100,
            # danger bas (il faut sauter)
            ennemi_plus_proche is not None and ennemi_plus_proche.rect.left <= 600 and ennemi_plus_proche.rect.bottom < 100,
            # danger haut (baisse toi)
            ennemi_plus_proche is not None and ennemi_plus_proche.rect.left <= 200,  # danger imminent
            not dino.isJumping and not dino.isDucking,  # dino avance droit
            dino.isJumping,  # le dino est en saut
            dino.isDucking  # le dino est accroupi
        ]

        for i in range(len(state)):
            if state[i]:
                state[i] = 1
            else:
                state[i] = 0

        return np.asarray(state)

    def set_reward(self, dino, cacti, pteras):
        self.reward = 0
        if dino.isDead:  # Punition pour avoir été tué
            self.reward = -10
            return self.reward
        ennemi_plus_proche = None
        for c in cacti:
            if c.rect.left <= 10:  # Rewards pour avoir dépassé un cactus / ptera
                self.reward = 10
        for p in pteras:
            if p.rect.left <= 10:
                self.reward = 10
        return self.reward

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay_new(self, memory, batch_size):
        if len(memory) > batch_size:
            minibatch = random.sample(memory, batch_size)
        else:
            minibatch = memory
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(np.array([next_state]))[0])
            target_f = self.model.predict(np.array([state]))
            target_f[0][np.argmax(action)] = target
            self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)

    def train_short_memory(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target = reward + self.gamma * np.amax(self.model.predict(next_state.reshape((1, 6)))[0])
        target_f = self.model.predict(state.reshape((1, 6)))
        target_f[0][np.argmax(action)] = target
        self.model.fit(state.reshape((1, 6)), target_f, epochs=1, verbose=0)
