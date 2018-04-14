#https://keon.io/deep-q-learning/
if __name__ == "main":

    env = gym.make('Pendulum-v0')
    agent = DQNAgent(env)

    for e in range(episodes):

        state = env.reset()
        state  = np.reshape(state, [1, 3])

        for time_t in range(500):
            env.render()

            action = agent.act(state)

            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, 3])

            agent.remember(state, action, reward, next_state, done)

            state = next_state

            if done:
                print("episode: {}/{}, score; {}".format(e, episodes, time_t))

                break

            agent.replay(32)
