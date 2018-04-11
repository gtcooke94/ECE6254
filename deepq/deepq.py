#https://keon.io/deep-q-learning/
#https://towardsdatascience.com/reinforcement-learning-w-keras-openai-actor-critic-models-f084612cfd69

import keras
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K


EPISODES = 1000

class ActorCritic:
    def __init__(self, env, sess):
        self.env = env
        self.sess = sess

        self.learning_rate = 0.001
        self.epsilon_decay = .995
        self.gamma = .95
        self.tau   = .125
        self.memory = deque(maxlen=2000)

        self.actor_state_input, self.actor_model = self.create_actor_model()
        _, self.target_actor_model = self.create_actor_model()

        self.actor_critic_grad = tf.placeholder(tf.float32, [None, self.env.action_space.shape[0]])

        actor_model_weight = self.actor_model.trainable_weights
        self.actor_grads = tf.gradients(self.actor_model.output, actor_model_weights, -self.actor_critic_grad)
        grads = zip(self.actor_grads, actor_model_weights)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)

    def create_actor_model(self):
        state_input = Input(shape=self.env.observation_space.shape)
        h1 = Dense(24, activation='relu')(state_input)
        h2 = Dense(48, activation = 'relu')(h1)
        h3 = Dense(24, activation='relu')(h2)
        output = Dense(self.env.action_space.shape[0], activation='relu')(h3)

        model = Model(input=state_input, output=output)
        adam = Adam(lr=0.001)
        model.compile(loss='mse', optimizer=adam)
        return state_input, model

class DQNAgent:
    def __init__(self, state_size, action_size, action_low, action_high):
        self.state_size = state_size
        # self.action_size = action_size
        self.action_size = 1
        self.action_low = action_low
        self.action_high = action_high
        self.memory = deque(maxlen = 2000)

        # Discount Rate
        self.gamma = 0.95
        
        # Exploration rate
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            # return random.randrange(self.action_size)
            return random.uniform(self.action_low, self.action_high)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay



if __name__ == "__main__":

    env = gym.make('Pendulum-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    action_low = env.action_space.low
    action_high = env.action_space.high

    agent = DQNAgent(state_size, action_size, action_low, action_high)


    done = False
    batch_size = 32

    for e in range(EPISODES):

        state = env.reset()
        state  = np.reshape(state, [1, 3])

        for time_t in range(1000):
            env.render()

            action = agent.act(state)
            print(action)

            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, 3])

            agent.remember(state, action, reward, next_state, done)

            state = next_state

            if done:
                print("episode: {}/{}, score; {}".format(e, EPISODES, time_t))

                break

        if len(agent.memory) > batch_size:
            agent.replay(32)
