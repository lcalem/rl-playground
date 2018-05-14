import gym
import numpy as np
import random

env = gym.make("CartPole-v0")

# discounting
GAMMA = 0.99

# learning and exploration rates
ALPHA = 0.1
EPSILON = 0.01

# episodes
MAX_EPISODES = 10000
MAX_TIME = 200

# solving is defined by getting an average reward > 195.0 over 100 consecutive trials (https://github.com/openai/gym/wiki/CartPole-v0)
SOLVED_REWARD = 195
SOLVED_STREAK = 100

# environment (dimensions are cart_x, cart_x', pole_theta, pole_theta')
NB_BINS = 10  # arbitrary
NB_ACTIONS = env.action_space.n
OBS_BOUNDS = [(-2.4, 2.4), (-1, 1), (-2, 2), (-3.5, 3.5)]         # custom bounds found here (https://gist.github.com/heechul/9f8f43c229fc790af4a8f073108ed49f#file-inverted-qlearn-py-L106)
OBS_BINS = [np.linspace(min_bound, max_bound, NB_BINS) for min_bound, max_bound in OBS_BOUNDS]


def run():

    # (dimensions are cart_x, cart_x', pole_theta, pole_theta', actions)
    qvalue_table = np.zeros((NB_BINS, NB_BINS, NB_BINS, NB_BINS, NB_ACTIONS))

    current_streak = 0

    for i_episode in range(MAX_EPISODES):
        observation = env.reset()
        state_0 = discretize_state(observation)

        sum_rewards = 0

        for t in range(MAX_TIME):

            env.render()

            # take action
            # action = env.action_space.sample()
            action = select_action(state_0, qvalue_table)
            observation, reward, done, info = env.step(action)
            state_1 = discretize_state(observation)

            # update table
            try:
                qvalue_table[state_0 + (action,)] += ALPHA * (reward + GAMMA * max(qvalue_table[state_1]) - qvalue_table[state_0 + (action,)])
            except Exception as e:
                print("state0 %s, action %s" % (str(state_0), action))
                print("state1 %s, action %s" % (str(state_1), action))
                print("non discrete state %s" % str(observation))
                print("Previous value %s" % str(qvalue_table[state_0 + (action,)]))
                print("Best for next state %s" % str(max(qvalue_table[state_1])))
                raise

            # print(reward)
            sum_rewards += reward
            state_0 = state_1

            if done:
                print("Episode %s finished after %s timesteps (current streak %s)" % (i_episode, t + 1, current_streak))

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
    return tuple([min(np.digitize(x=[state[i]], bins=OBS_BINS[i])[0], NB_BINS - 1) for i in range(len(state))])


def select_action(state, qvalue_table):
    '''
    state is a discretized state
    epsilon-greedy
    '''
    if random.random() < EPSILON:
        action = env.action_space.sample()
    else:
        action = np.argmax(qvalue_table[state])

    return action


if __name__ == "__main__":
    print(env.observation_space)
    print(env.observation_space.low)
    print(env.observation_space.high)
    run()
