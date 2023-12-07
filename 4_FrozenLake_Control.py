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
epsilon = 0.1

def play_episode_prediction(q_values):
    state = env.reset()[0]
    done = False
    r_s = []
    while not done:
        action = np.argmax(q_values[state])
        state, reward, done, _, _ = env.step(action)
        r_s.append(reward)
    return r_s

def learn_q_table():

    state = env.reset()[0]
    action = random.randint(0, 3)
    done = False

    r_s = []

    while not done:

        next_state, reward, done, _, _ = env.step(action)
        next_action = random.randint(0, 3)

        q_values[state, action] += alpha*(reward + q_values[next_state, next_action] - q_values[state, action])
        state = next_state
        action = next_action

        r_s.append(reward)

    return r_s


def play_episode(policy=None):
    state = env.reset()[0]
    done = False
    r_s = []
    while not done:
        if policy is None:
            action = random.randint(0, 3)
        else:
            winners = np.flatnonzero(policy[state] == np.max(policy[state]))
            probility = []
            for a in actions:
                if a in winners:
                    val = (epsilon / no_actions) + (1 - epsilon) / len(winners)
                    probility.append(val)
                else:
                    probility.append(epsilon / no_actions)
            
            action = np.random.choice(no_actions, 1, p=probility)[0]

        prev_state = state
        state, reward, done, _, _ = env.step(action)
        if policy is not None:
            q_values[prev_state][action] += alpha * (reward + np.max(q_values[state]) - q_values[prev_state][action])
        r_s.append(reward)
    return r_s



def main():
    # learn q values
    # successful_episodes = 1000
    # while successful_episodes > 0:
    #     r_s = learn_q_table()
    #     if sum(r_s) > 0:
    #         successful_episodes -= 1


    no_episodes = 1000
    rewards_random = []
    for i in range(0, no_episodes):
        r = play_episode()
        rewards_random.append(sum(r))

        # TODO: update q-values with MC-prediction

    plot_data_rand = np.cumsum(rewards_random)

    rewards = []
    for i in range(0, no_episodes):
        r = play_episode(q_values)
        rewards.append(sum(r))

        # TODO: update q-values with MC-prediction

    plot_data = np.cumsum(rewards)

    # plot the rewards
    plt.figure()
    plt.xlabel("No. of episodes")
    plt.ylabel("Sum of Rewards")
    plt.plot(range(0, no_episodes), plot_data_rand, label="random")
    plt.plot(range(0, no_episodes), plot_data, label="epsilon")
    plt.legend()
    plt.show()


main()
