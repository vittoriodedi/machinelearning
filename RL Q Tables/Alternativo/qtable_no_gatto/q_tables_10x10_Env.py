from os import path
from typing import Optional

import numpy as np
import random

import gymnasium as gym
from gymnasium import Env, spaces
from gymnasium.envs.toy_text.utils import categorical_sample

import pygame
import matplotlib.pyplot as plt
import pickle

gym.register(
    id="q_tables_10x10_Env",
    entry_point="q_tables_10x10_Env:cat_vs_mouse_Env",
    kwargs={"map_name": "10x10"},
    max_episode_steps=200,
    reward_threshold=0.91,
)

#======================================ENVIRONMENT======================================
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

MAPS = {
   "10x10": [
        "SPPPPPPPPP",
        "PTPPTTPPTP",
        "PPTPPPPPPP",
        "PPPPPPPPPP",
        "PPPTPPTPPP",
        "PPPTPPTPPP",
        "PPPPPPPPPP",
        "PPTPPPPTPP",
        "PTPPTTPPTP",
        "PPPPPPPPPF"
    ],
}



class cat_vs_mouse_Env(Env):
    metadata = {
        "render_modes": ["human", "ansi", "rgb_array"],
        "render_fps": 4,
    }

    def __init__( self, render_mode: Optional[str] = None, desc=None, map_name="10x10"):
        desc = MAPS[map_name]
        self.desc = desc = np.asarray(desc, dtype="c")
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.reward_range = (-1, 1)

        number_of_actions = 4
        number_of_states = nrow * ncol

        self.initial_state_distrib = np.array(desc == b"S").astype("float64").ravel()
        self.initial_state_distrib /= self.initial_state_distrib.sum()

        self.P = {s: {a: [] for a in range(number_of_actions)} for s in range(number_of_states)}

        def to_state(row, col):
            return row * ncol + col

        def inc(row, col, act):
            if act == LEFT:
                col = max(col - 1, 0)
            elif act == DOWN:
                row = min(row + 1, nrow - 1)
            elif act == RIGHT:
                col = min(col + 1, ncol - 1)
            elif act == UP:
                row = max(row - 1, 0)
            return (row, col)

        def update_probability_matrix(row, col, action):
            newrow, newcol = inc(row, col, action)
            newstate = to_state(newrow, newcol)
            newletter = desc[newrow, newcol]
            terminated = bytes(newletter) in b"FT"
            reward = float(newletter == b"F")
            return newstate, reward, terminated

        for row in range(nrow):
            for col in range(ncol):
                s = to_state(row, col)
                for a in range(4):
                    li = self.P[s][a]
                    letter = desc[row, col]
                    if letter in b"T":
                        li.append((-1, s, 0, True))
                    elif letter in b"F":
                        li.append((1, s, 0, True))
                    else:
                        li.append((1.0, *update_probability_matrix(row, col, a)))

        self.observation_space = spaces.Discrete(number_of_states)
        self.action_space = spaces.Discrete(number_of_actions)

        self.render_mode = render_mode

        # pygame utils
        self.window_size = 10,10
        self.cell_size = (self.window_size[0] // self.ncol, self.window_size[1] // self.nrow)
        self.window_surface = None
        self.clock = None
        self.open_trap_img = None
        self.closed_trap_img = None
        self.floor_img = None
        self.mouse_images = None
        self.goal_img = None
        self.start_img = None

        # Additional variables
        self.q_table = None
        self.episode = '---'
        self.pygame_initialized = False    # flag to determine if pygame has been initialized
        self.text_padding = 5
        self.show_q_table = True

    def step(self, a):
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, t = transitions[i]
        self.s = s
        self.lastaction = a

        if self.pygame_initialized:

            for event in pygame.event.get():
                if(event.type == pygame.KEYDOWN):

                    if(event.key == pygame.K_ESCAPE):
                        pygame.quit()
                        import sys;
                        sys.exit()

                    elif(event.key == pygame.K_3):
                        self.metadata["render_fps"]+=10

                    elif(event.key == pygame.K_2):
                        self.metadata["render_fps"]-=10
                        if(self.metadata["render_fps"]<=0):
                            self.metadata["render_fps"]=1

                    elif(event.key == pygame.K_1):
                        self.metadata["render_fps"]=0

                    elif(event.key == pygame.K_0):
                        self.metadata["render_fps"]=4

                    elif(event.key == pygame.K_9):
                        self.render_mode=None if(self.render_mode=="human") else "human"
                    
                    elif(event.key == pygame.K_s):
                        self.show_q_table = not self.show_q_table
                        

                if(event.type == pygame.QUIT):
                    pygame.quit()
                    import sys;
                    sys.exit()

        if self.render_mode == "human":
            self.render()
        return (int(s), r, t, False, {"prob": p})

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.s = categorical_sample(self.initial_state_distrib, self.np_random)
        self.lastaction = None
        
        if self.render_mode == "human":
            self.render()
        return int(self.s), {"prob": 1}


    def render(self):
        return self._render_gui(self.render_mode)


    def _render_gui(self, render_mode):

        if self.window_surface is None:
            if not pygame.get_init():
                pygame.init()
            self.pygame_initialized = True

            # fonts for UI
            self.ui_font = pygame.font.SysFont("Courier",20)
            self.q_font = pygame.font.SysFont("Courier",9)
            self.q_font_bold = pygame.font.SysFont("Courier",10,True)

            pygame.display.init()
            pygame.display.set_caption("Mouse vs cat")
            # self.window_surface = pygame.display.set_mode(self.window_size)

            self.window_surface = pygame.display.set_mode((1300, 700))
            self.display_width, display_height = 700, 700   #dimensione mappa

            self.grid_size = display_height
            if self.grid_size > self.display_width:
                self.grid_size = self.display_width

            self.window_size = 700, 700
            self.cell_size = (self.window_size[0] // self.ncol, self.window_size[1] // self.nrow,)

        if self.clock is None:
            self.clock = pygame.time.Clock()
        if self.open_trap_img is None:
            file_name = path.join(path.dirname(__file__), "img/open_trap.png")
            self.open_trap_img = pygame.transform.scale(pygame.image.load(file_name), self.cell_size)
        if self.closed_trap_img is None:
            file_name = path.join(path.dirname(__file__), "img/closed_trap.png")
            self.closed_trap_img = pygame.transform.scale(pygame.image.load(file_name), self.cell_size)
        if self.floor_img is None:
            file_name = path.join(path.dirname(__file__), "img/floor.png")
            self.floor_img = pygame.transform.scale(pygame.image.load(file_name), self.cell_size)
        if self.goal_img is None:
            file_name = path.join(path.dirname(__file__), "img/goal.png")
            self.goal_img = pygame.transform.scale(pygame.image.load(file_name), self.cell_size)
        if self.start_img is None:
            file_name = path.join(path.dirname(__file__), "img/hole.png")
            self.start_img = pygame.transform.scale(pygame.image.load(file_name), self.cell_size)
        if self.mouse_images is None:
            mouses = [
                path.join(path.dirname(__file__), "img/mouse_left.png"),
                path.join(path.dirname(__file__), "img/mouse_down.png"),
                path.join(path.dirname(__file__), "img/mouse_right.png"),
                path.join(path.dirname(__file__), "img/mouse_up.png"),
            ]
            self.mouse_images = [
                pygame.transform.scale(pygame.image.load(f_name), self.cell_size)
                for f_name in mouses
            ]

        self.window_surface.fill((255,255,255))

        desc = self.desc.tolist()

        for y in range(self.nrow):
            for x in range(self.ncol):
                pos = (x * self.cell_size[0], y * self.cell_size[1])
                rect = (*pos, *self.cell_size)

                self.window_surface.blit(self.floor_img, pos)
                if desc[y][x] == b"T":
                    self.window_surface.blit(self.open_trap_img, pos)
                elif desc[y][x] == b"F":
                    self.window_surface.blit(self.goal_img, pos)
                elif desc[y][x] == b"S":
                    self.window_surface.blit(self.start_img, pos)

                pygame.draw.rect(self.window_surface, (200, 200, 200), rect, 1)



                # render q values
                if(self.show_q_table):
                    if(self.q_table is not None):

                        # map x y location to q table state
                        state = self.nrow * y + x

                        # Get index of max Q value
                        max_q_idx = np.argmax(self.q_table[state])
                        # If max Q value is 0, it hasn't receive learning yet, ignore
                        if(self.q_table[state][max_q_idx]==0):
                            max_q_idx = -1



                        # Calculate position of Q values within a cell to be displayed.
                        # Create a dummy render of a q value to help with positioning calculation.
                        q_img = self.q_font.render(".0000", True, (0,0,0), (255,255,255))
                        q_img_x = q_img.get_width()
                        q_img_y = q_img.get_height()
                        q_pos = [
                            (pos[0]+self.text_padding,
                                pos[1]+self.cell_size[1]/2), # left
                            (pos[0]+self.cell_size[0]/2-q_img_x/2,
                                pos[1]+self.cell_size[1] - self.text_padding - q_img_y), # bottom
                            (pos[0]+self.cell_size[0]-self.text_padding-q_img_x,
                                pos[1]+self.cell_size[1]/2), # right
                            (pos[0]+self.cell_size[0]/2-q_img_x/2,
                                pos[1]+self.text_padding), # top
                        ]

                        # Loop thru the 4 Q values for the current state
                        for i in range(4):
                            # Format q for display
                            q = '{:.4f}'.format(self.q_table[state][i]).lstrip('0')

                            if(max_q_idx == i):
                                # Render q in bold font. Render takes (value, antialias, color, background color)
                                q_img = self.q_font_bold.render(q, True, (102, 153, 0))
                            else:
                                # Render q in regular font
                                q_img = self.q_font.render(q, True, (50,50,0))

                            # Display text img at position
                            self.window_surface.blit(q_img, q_pos[i])




        # paint the mouse
        bot_row, bot_col = self.s // self.ncol, self.s % self.ncol
        cell_rect = (bot_col * self.cell_size[0], bot_row * self.cell_size[1])
        last_action = self.lastaction if self.lastaction is not None else 1
        mouse_img = self.mouse_images[last_action]

        if desc[bot_row][bot_col] == b"T":
            self.window_surface.blit(self.closed_trap_img, cell_rect)
        else:
            self.window_surface.blit(mouse_img, cell_rect)


        # render episodes
        text_img = self.ui_font.render("Episode: " + str(self.episode), True, (0,0,0), (255,255,255))
        text_pos = (self.grid_size + self.text_padding, self.text_padding)
        self.window_surface.blit(text_img, text_pos)

        # render shortcut keys
        text_lines = ["azioni:", 
                        "s : mostra/nascondi Q values",                         
                        "0 : Reset FPS",
                        "1 : FPS infiniti",
                        "2 : Diminuisce FPS",
                        "3 : Aumenta FPS",
                        "9 : attiva/disattiva render mode",
                        " ",
                        "ESC chiudi finestra e",
                        "perde training",
        ]

        starting_y = text_img.get_height() + text_img.get_height()
        text_line_height = text_img.get_height()
        for i, line in enumerate(text_lines):
            text_img = self.ui_font.render(line, True, (0,0,0), (255,255,255))
            text_pos = (self.grid_size + self.text_padding, starting_y + i*text_line_height + self.text_padding)
            self.window_surface.blit(text_img, text_pos)

        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window_surface is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()

    def set_q(self, q_tablee):
        self.q_table = q_tablee

    def set_episode(self, episodee):
        self.episode = episodee

'''    
    0  1  2  3  4  5  6  7  8  9
    10 11 12 13 14 15 16 17 18 19
    20 21 22 23 24 25 26 27 28 29
    30 31 32 33 34 35 36 37 38 39
    40 41 42 43 44 45 46 47 48 49
    50 51 52 53 54 55 56 57 58 59
    60 61 62 63 64 65 66 67 68 69
    70 71 72 73 74 75 76 77 78 79
    80 81 82 83 84 85 86 87 88 89
    90 91 92 93 94 95 96 97 98 99
'''
#======================================ENVIRONMENT_END======================================



#======================================EXECUTE======================================


def run(episodes, is_training=True, render=False):

    env = gym.make('q_tables_10x10_Env', desc=None, map_name="10x10", render_mode='human' if render else None)
    if(is_training):
        q = np.zeros((env.observation_space.n, env.action_space.n))
    else:
        f = open('mouse_no_cat10x10.pkl', 'rb')
        q = pickle.load(f)
        f.close()

    learning_rate_a = 0.9 # alpha or learning rate
    discount_factor_g = 0.9 # gamma or discount rate. Near 0: more weight/reward placed on immediate state. Near 1: more on future state.
    epsilon = 1         # 1 = 100% random actions
    epsilon_decay_rate = 0.00014        # epsilon decay rate. 1/0.00014 = 7150 circa
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
    print("Sum of rewards per 100 episodes: ", sum_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Sum of rewards per 100 episodes')
    plt.title("10x10 traps no cat")
    plt.savefig('mouse_no_cat10x10.png')

    #grafico che fa un punto ogni 100 episodi, e il punto è la somma delle ricompense dei 100 episodi precedenti

    if is_training:
        f = open("mouse_no_cat10x10.pkl","wb")
        pickle.dump(q, f)
        f.close()

#======================================EXECUTE_END======================================

run(10000, is_training=False, render=True)
exit(1)
