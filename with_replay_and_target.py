import gym
import random
import numpy as np

from collections import deque

from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

EPISODES = 1000


class DQNAgent:
    def __init__(self, environment):
        self.env = environment

        self.state_size = environment.observation_space.shape[0]
        self.action_size = environment.action_space.n

        self.memory = deque(maxlen=20000)

        self.gamma = 0.99
        self.learning_rate = 0.001

        self.epsilon = 0.2
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.95

        self.train_network = self.__create_network()

        self.target_network = self.__create_network()
        self.target_network.set_weights(self.train_network.get_weights())

    def __create_network(self):
        model = Sequential()

        model.add(Dense(24, activation='relu', input_dim=self.state_size))
        model.add(Dense(48, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        return model

    def choose_action(self, state):
        self.epsilon = max(self.epsilon_min, self.epsilon)

        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        return np.argmax(self.train_network.predict(state))

    def remember(self, state, action, next_state, reward, done):
        self.memory.append([state, action, next_state, reward, done])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        # Get batch from memory
        mini_batch = random.sample(self.memory, batch_size)

        # Get states tables
        states = []
        next_states = []

        for state, action, new_state, reward, done in mini_batch:
            states.append(state)
            next_states.append(new_state)

        states = np.reshape(states, [batch_size, 2])
        next_states = np.reshape(next_states, [batch_size, 2])

        # Predict target values
        state_targets = self.train_network.predict(states)
        next_state_targets = self.target_network.predict(next_states)

        # Train neural network
        i = 0
        for state, action, new_state, reward, done in mini_batch:
            state_targets[i][action] = reward if done else reward + self.gamma * max(next_state_targets[i])
            i += 1

        self.train_network.fit(states, state_targets, epochs=1, verbose=0)

    def start(self, batch_size):
        for episode in range(EPISODES):
            state = env.reset()
            state = np.reshape(state, [1, self.state_size])

            total_reward = 0

            while True:
                action = self.choose_action(state)

                next_state, reward, done, _ = env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])

                self.remember(state, action, next_state, reward, done)
                self.replay(batch_size)

                state = next_state
                total_reward += reward

                if done:
                    break

                if episode % 100 == 0:
                    env.render()

            # Update epsilon for increase correct step probabilities
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            # Save train network parameters
            if total_reward > -200:
                #self.trainNetwork.save('./trainNetworkInEPS{}.h5'.format(episode))
                print("Success in epsoide {}! Epsilon: {} and Reward: {}".format(episode, self.epsilon, total_reward))

            self.target_network.set_weights(self.train_network.get_weights())


if __name__ == '__main__':
    env = gym.make('MountainCar-v0')

    agent = DQNAgent(env)
    agent.start(batch_size=32)
