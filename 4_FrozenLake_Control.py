import random
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

env = gym.make("FrozenLake-v1")
random.seed(0)
np.random.seed(0)

print("## Frozen Lake ##")

no_states = env.observation_space.n
no_actions = env.action_space.n
actions = range(0, env.unwrapped.action_space.n)
q_values = np.zeros((no_states, no_actions))
q_counter = np.zeros((no_states, no_actions))
alpha = 0.5


def play_episode(q_values, epsilon):

    state, _ = env.reset(seed=0)
    action = choose_action(q_values, state, epsilon)
    done = False

    r_s = []
    while not done:
        next_state, reward, done, _, _ = env.step(action)
        next_action = choose_action(q_values, next_state, epsilon)

        q_values[state, action] += alpha*(reward + np.max(q_values[next_state]) - q_values[state, action])
        state = next_state
        action = next_action

        r_s.append(reward)

    return r_s


def choose_action(q_values, state, epsilon):
    if random.random() > epsilon:
        # note: there can be more than one best action, e.g. [0, 0, 0, 0]
        # problem: np.argmax(q_values[state]) will always return the first max element
        # better: find all max elements and select one randomly
        max_indices = [i for i, v in enumerate(q_values[state]) if v == max(q_values[state])]
        return random.choice(max_indices)
    else:
        return random.randint(0, 3)


def main():
    # learn q values
    # successful_episodes = 1000
    # while successful_episodes > 0:
    #     r_s = learn_q_table()
    #     if sum(r_s) > 0:
    #         successful_episodes -= 1


    no_episodes = 1000
    epsilons = [0.01, 0.1, 0.5, 1.0]

    plot_data = []
    for e in epsilons:

        q_values = np.zeros((no_states, no_actions))

        rewards = []
        for j in range(0, no_episodes):
            r = play_episode(q_values, epsilon=e)
            rewards.append(sum(r))

        plot_data.append(np.cumsum(rewards))

    # plot the rewards
    plt.figure()
    plt.xlabel("No. of episodes")
    plt.ylabel("Total reward")
    for i, eps in enumerate(epsilons):
        plt.plot(range(0, no_episodes), plot_data[i], label="e=" + str(eps))
    plt.legend()
    plt.show()


main()
