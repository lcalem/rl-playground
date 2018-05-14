import gym
import numpy as np
import math
import random

env = gym.make("CartPole-v0")

# discounting
GAMMA = 0.99

# episodes
MAX_EPISODES = 3000
MAX_TIME = 200

# solving is defined by getting an average reward > 195.0 over 100 consecutive trials (https://github.com/openai/gym/wiki/CartPole-v0)
SOLVED_REWARD = 195
SOLVED_STREAK = 100

# environment (dimensions are cart_x, cart_x', pole_theta, pole_theta')
NB_DIMS = 4
NB_ACTIONS = env.action_space.n
NB_BINS = (2, 2, 10, 5)  # arbitrary for theta and theta' only

# reduce the biggest bounds
OBS_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))
OBS_BOUNDS[1] = [-0.5, 0.5]
OBS_BOUNDS[3] = [-1, 1]

OBS_BINS = [np.linspace(OBS_BOUNDS[i][0], OBS_BOUNDS[i][1], NB_BINS[i] - 1) for i in range(NB_DIMS)]


def run():

    # (dimensions are cart_x, cart_x', pole_theta, pole_theta', actions)
    qvalue_table = np.zeros(NB_BINS + (NB_ACTIONS,))
    current_streak = 0

    for i_episode in range(MAX_EPISODES):
        observation = env.reset()
        state_0 = discretize_state(observation)
        epsilon = max(0.01, min(1, 1.0 - math.log10((i_episode + 1) / 25)))
        alpha = max(0.1, min(0.5, 1.0 - math.log10((i_episode + 1) / 25)))

        # print("epsilon %s, alpha %s" % (epsilon, alpha))
        sum_rewards = 0

        for t in range(MAX_TIME):

            env.render()

            # take action
            action = select_action(state_0, qvalue_table, epsilon)
            observation, reward, done, info = env.step(action)
            state_1 = discretize_state(observation)

            # update table
            try:
                qvalue_table[state_0 + (action,)] += alpha * (reward + GAMMA * max(qvalue_table[state_1]) - qvalue_table[state_0 + (action,)])
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
                if sum_rewards > SOLVED_REWARD:
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

    assert len(state) == NB_DIMS
    state_repr = [0] * NB_DIMS
    for i in range(NB_DIMS):
        if state[i] <= OBS_BOUNDS[i][0]:
            val = 0
        elif state[i] >= OBS_BOUNDS[i][1]:
            val = NB_BINS[i] - 1
        else:
            val = max(np.digitize(x=[state[i]], bins=OBS_BINS[i])[0] - 1, 0)

        state_repr[i] = val

    return tuple(state_repr)

    # return tuple([min(np.digitize(x=[state[i]], bins=OBS_BINS[i])[0] - 1, NB_BINS[i] - 1) for i in range(len(state))])


def select_action(state, qvalue_table, epsilon):
    '''
    state is a discretized state
    epsilon-greedy
    '''
    if random.random() < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(qvalue_table[state])

    return action


if __name__ == "__main__":
    print(env.observation_space)
    print(env.observation_space.low)
    print(env.observation_space.high)
    run()
