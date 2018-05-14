import gym

env = gym.make("CartPole-v0")

# discounting
GAMMA = 0.99

# learning and exploration rates
ALPHA = 0.1
EPSILON = 0.01

# episodes
MAX_EPISODES = 10000
MAX_TIME = 1000

# solving is defined by getting an average reward > 195.0 over 100 consecutive trials (https://github.com/openai/gym/wiki/CartPole-v0)
SOLVED_REWARD = 195
SOLVED_STREAK = 100


def run():

        current_streak = 0

            for i_episode in range(MAX_EPISODES):
                        observation = env.reset()

                                sum_rewards = 0

                                        for t in range(MAX_TIME):

                                                        env.render()
                                                                    # print(observation)

                                                                                action = env.action_space.sample()
                                                                                            observation, reward, done, info = env.step(action)

                                                                                                        # print(reward)
                                                                                                                    sum_rewards += reward

                                                                                                                                if done:
                                                                                                                                                    print("Episode %s finished after %s timesteps (current steak %s)" % (i_episode, t + 1, current_streak))

                                                                                                                                                                    # episode success
                                                                                                                                                                                    if sum_rewards / t > SOLVED_REWARD:
                                                                                                                                                                                                            current_streak += 1
                                                                                                                                                                                                                            else:
                                                                                                                                                                                                                                                    current_streak = 0

                                                                                                                                                                                                                                                                    break

                                                                                                                                                                                                                                                                        if current_streak >= SOLVED_STREAK:
                                                                                                                                                                                                                                                                                        print("Problem solved! In %s episodes" % i_episode)
                                                                                                                                                                                                                                                                                                    break

                                                                                                                                                                                                                                                                                                    else:
                                                                                                                                                                                                                                                                                                                print(":(")


                                                                                                                                                                                                                                                                                                                        def discretize_state(state):
                                                                                                                                                                                                                                                                                                                            '''
                                                                                                                                                                                                                                                                                                                                state is the environment raw response
                                                                                                                                                                                                                                                                                                                                    each of the 4 dimensions is discretized into appropriate bins

                                                                                                                                                                                                                                                                                                                                        '''
                                                                                                                                                                                                                                                                                                                                            pass

                                                                                                                                                                                                                                                                                                                                            if __name__ == "__main__":
                                                                                                                                                                                                                                                                                                                                                run()

