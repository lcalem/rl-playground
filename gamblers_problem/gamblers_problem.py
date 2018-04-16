import matplotlib.pyplot as plt

from pprint import pprint

P_HEAD = 0.25
THRESHOLD = 1e-8
MAX_STEPS = 600
DISCOUNT = 1
STATES = range(0, 101)  # 0 to 100 included


# policy: mapping from capital to stake

def action_return(state_values, s, a):
    # threre is only 2 successor states, either it's head and the gambler wins his stake, or tail and he looses the stake
    win = P_HEAD * (DISCOUNT * state_values[min(100, s + a)])
    loose = (1 - P_HEAD) * (DISCOUNT * state_values[s - a])

    return win + loose


def value_iteration():
    delta = 100000
    step_count = 0
    state_values = {s: 0 for s in STATES}
    state_values[100] = 1.0  # win!

    while (delta > THRESHOLD) or (step_count > MAX_STEPS):

        for state in STATES[1:100]:
            possible_actions = range(1, min(state, 100 - state) + 1)  # +1 because you can bet all of your current capital
            print("for state %s, possible actions are %s" % (state, str(possible_actions)))

            v = state_values[state]
            v_s = max([action_return(state_values, state, a) for a in possible_actions])
            state_values[state] = v_s
            delta = abs(v - v_s)

        step_count += 1

    pprint(state_values)
    return state_values


def compute_policy(state_values):
    policy = [0] * 101

    for state in STATES[1:100]:
        possible_actions = range(1, min(state, 100 - state) + 1)
        action_returns = [action_return(state_values, state, a) for a in possible_actions]

        policy[state] = action_returns.index(max(action_returns))   # argmax

    return policy


def main():
    state_values = value_iteration()
    policy = compute_policy(state_values)

    # display
    plt.figure(1)
    plt.subplot(211)
    plt.scatter(STATES, policy)
    plt.title('policy')
    plt.xlabel('capital')
    plt.ylabel('stake')

    plt.subplot(212)
    fixed_k = sorted(state_values.keys())
    fixed_v = [state_values[k] for k in fixed_k]
    plt.scatter(fixed_k, fixed_v)
    plt.title('value estimates')
    plt.xlabel('capital')
    plt.ylabel('V(s)')
    plt.show()


if __name__ == '__main__':
    main()
