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
q_counter = np.zeros((no_states, no_actions))


def play_episode(q_values=None):

    state = env.reset()[0]
    done = False
    r_s = []
    s_a = []
    while not done:
        if q_values is None:
            action = random.randint(0, 3)
        else:
            action = np.argmax(q_values[state]) # choose greedy

        s_a.append((state, action))
        state, reward, done, _, _ = env.step(action)
        r_s.append(reward)
    return s_a, r_s

def main():
    successful_episodes = 1000
    while successful_episodes > 0:
        s_a, r_s = play_episode()

        # update q-values with MC-prediction
        for i, (s,a) in enumerate(s_a):
            return_i = sum(r_s[i:])
            q_counter[s][a] += 1
            q_values[s][a] += 1/q_counter[s][a] * (return_i - q_values[s][a])

        if sum(r_s) > 0:
            all_rewards = 0
            for i in range(0, 100):
                s_a, rewards = play_episode(q_values)
                all_rewards += sum(rewards)
            print(all_rewards / 100)
            successful_episodes -= 1

main()
