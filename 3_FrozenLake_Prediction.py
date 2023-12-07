import random
import gymnasium as gym
import numpy as np

env = gym.make("FrozenLake-v1")
random.seed(0)
np.random.seed(0)

print("## Frozen Lake ##")

no_states = env.observation_space.n
no_actions = env.action_space.n

q_values = np.random.rand(no_states, no_actions)
alpha = 0.01
epsilon = 0.4

def play_episode(policy=None):
    state = env.reset()[0]
    done = False
    r_s = []
    while not done:
        if policy is None:
            action = random.randint(0, 3)
        else:
            winners = np.flatnonzero(policy[state] == np.max(policy[state]))
            actions = []
            for a in no_actions:
                if a in winners:
                    val = (epsilon / no_actions) + (1 - epsilon) / len(winners)
                    actions.append(val)
                else:
                    actions.append(epsilon / no_actions)



        prev_state = state
        state, reward, done, _, _ = env.step(action)
        q_values[prev_state][action] += 0.01 * (reward + np.max(q_values[state]) - q_values[prev_state][action])
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
        success = play_episode()
        if(sum(success) > 0):
            successful_episodes -= 1
            

    all_rewards = 0
    for i in range(0, 100):
        rewards = play_episode(q_values)
        all_rewards += sum(rewards)
    print(q_values)
    print(all_rewards / 100)
main()
