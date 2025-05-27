import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

gym.register(
    id="q_tables_10x10_Env",
    entry_point="q_tables_10x10_Env:cat_vs_mouse_Env",
    kwargs={"map_name": "10x10"},
    max_episode_steps=200,
    reward_threshold=0.91,
)

def run(episodes, is_training=True, render=False):

    env = gym.make('q_tables_10x10_Env', desc=None, map_name="10x10", render_mode='human' if render else None)

    if(is_training):
        q = np.zeros((env.observation_space.n, env.action_space.n))
    else:
        f = open('mouse_si_cat10x10.pkl', 'rb')
        q = pickle.load(f)
        f.close()

    learning_rate_a = 0.9 # alpha or learning rate
    discount_factor_g = 0.9 # gamma or discount rate. Near 0: more weight/reward placed on immediate state. Near 1: more on future state.
    epsilon = 1         # 1 = 100% random actions
    epsilon_decay_rate = 0.00012        # epsilon decay rate. 1/0.0002 = 5,000
    rng = np.random.default_rng()   # random number generator

    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        print("Episode: ", i)
        state = env.reset()[0]  # states: 0 to 99, 0=top left corner,99=bottom right corner
        terminated = False      # True when fall in killed or reached goal
        truncated = False       # True when actions > 200


        while(not terminated and not truncated):
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample() # actions: 0=left,1=down,2=right,3=up
            else:
                action = np.argmax(q[state,:])

            new_state,reward,terminated,truncated,_ = env.step(action)

            # Check if the agent has been killed by a trap
            if terminated and reward == 0:
                reward = -1
                for a in range(env.action_space.n):
                    if a != action:
                        q[state, a] += learning_rate_a * 0.0001  # Increase by a small value


            if is_training:
                q[state, action] = q[state, action] + learning_rate_a * (reward + discount_factor_g * np.max(q[new_state, :]) - q[state, action])
            
                #print("State: ", state, "Action: ", action, "Reward: ", reward, "New State: ", new_state, "Q: ", q[state, action])

            # pass the q table and episode count to the environment for rendering
            if(env.render_mode=='human'):
                env.unwrapped.set_q(q)
                env.unwrapped.set_episode(i)

            state = new_state
            
            #muore se gatto su topo
            if state == env.unwrapped.get_cat_state():
                print("Mouse killed by cat")
                break

        epsilon = max(epsilon - epsilon_decay_rate, 0)

        if(epsilon==0):
            learning_rate_a = 0.0001

        if reward == 1:
            rewards_per_episode[i] = 1

    env.close()

    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
    plt.plot(sum_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Sum of rewards per 100 episodes')
    plt.title("10x10 traps with cat")
    plt.savefig('mouse_si_cat10x10.png')
    #grafico che fa un punto ogni 100 episodi, e il punto è la somma delle ricompense dei 100 episodi precedenti

    if is_training:
        f = open("mouse_si_cat10x10.pkl","wb")
        pickle.dump(q, f)
        f.close()

run(10000, is_training=True, render=True)