import gym
import random
import numpy as np

from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

EPISODES = 1000


class DQNAgent:
    def __init__(self, environment):
        self.env = environment

        self.state_size = environment.observation_space.shape[0]
        self.action_size = environment.action_space.n

        self.gamma = 0.99
        self.learning_rate = 0.001

        self.epsilon = 0.2
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.95

        self.train_network = self.__create_network()

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

    def play(self, state, action, next_state, reward, done):
        # Update neural network
        state_target = self.train_network.predict(state)
        state_target[0][action] = reward if done else reward + self.gamma * max(state_target[0])

        self.train_network.fit(state, state_target, epochs=1, verbose=0)

    def start(self):
        for episode in range(EPISODES):
            state = env.reset()
            state = np.reshape(state, [1, self.state_size])

            total_reward = 0

            while True:
                action = self.choose_action(state)

                next_state, reward, done, _ = env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])

                self.play(state, action, next_state, reward, done)

                # Update values for next state
                state = next_state
                total_reward += reward

                if done:
                    break

                if episode % 20 == 0:
                    env.render()

            # Update epsilon for increase correct step probabilities
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            # Save train network parameters
            if total_reward > -200:
                # self.trainNetwork.save('./trainNetworkInEPS{}.h5'.format(episode))
                print("Success in epsoide {}! Epsilon: {} and Reward: {}".format(episode, self.epsilon, total_reward))


if __name__ == '__main__':
    env = gym.make('MountainCar-v0')

    agent = DQNAgent(env)
    agent.start()
