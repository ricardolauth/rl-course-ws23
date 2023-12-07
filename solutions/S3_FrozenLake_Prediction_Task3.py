import random
import gymnasium as gym
import numpy as np

env = gym.make("FrozenLake-v1")
random.seed(0)
np.random.seed(0)

print("## Frozen Lake ##")

no_states = env.observation_space.n
no_actions = env.action_space.n

q_values = np.zeros((no_states, no_actions))

alpha = 0.01

def play_episode(q_values):

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


def main():
    successful_episodes = 1000
    while successful_episodes > 0:
        r_s = learn_q_table()

        if sum(r_s) > 0:
            #print(q_values)
            all_rewards = 0
            for i in range(0, 100):
                rewards = play_episode(q_values)
                all_rewards += sum(rewards)
            print(all_rewards / 100)
            successful_episodes -= 1

main()
