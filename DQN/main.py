import time
import pygame
import random
import torch
import numpy as np
import torch.nn as nn
import gymnasium as gym
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt

class CatMouseCheeseEnv(gym.Env):
    def __init__(self, grid_size=10):
        super(CatMouseCheeseEnv, self).__init__()
        self.grid_size = grid_size
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=0, high=grid_size-1, shape=(6,), dtype=np.int32)
        self.walls = self._generate_walls()
        self.reset()

    def _generate_walls(self):
        walls = {(i, j): False for i in range(self.grid_size) for j in range(self.grid_size)}
        static_walls = [(1, 1),(2, 2),(1, 4),(1, 5),(2, 7),(1, 8),(4, 3),(5, 3),(4, 6),(5, 6),(7, 2),(8, 1),(8, 4),(8, 5),(7, 7),(8, 8)]
        for wall in static_walls:
            walls[wall] = True
        return walls

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.mouse_pos = [0, 0]
        self.cheese_pos = [9, 9]

        while True:
            self.cat_pos = [random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)]
            if self.cat_pos != self.mouse_pos and self.cat_pos != self.cheese_pos and not self.walls[tuple(self.cat_pos)]:
                break

        state = np.array([*self.mouse_pos, *self.cat_pos, *self.cheese_pos], dtype=np.int32)
        return state, {}

    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def _can_move(self, pos, action):
        i, j = pos
        if action == 0 and (i == 0 or self.walls[(i-1, j)]):
            return False
        if action == 1 and (i == self.grid_size - 1 or self.walls[(i+1, j)]):
            return False
        if action == 2 and (j == 0 or self.walls[(i, j-1)]):
            return False
        if action == 3 and (j == self.grid_size - 1 or self.walls[(i, j+1)]):
            return False
        return True

    def step(self, action):
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

        if self._can_move(self.mouse_pos, action):
            return False
        return True

    def step(self, action):
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

        if self._can_move(self.mouse_pos, action):
            self.mouse_pos = [self.mouse_pos[0] + moves[action][0], self.mouse_pos[1] + moves[action][1]]

        valid_cat_moves = [a for a in moves.keys() if self._can_move(self.cat_pos, a)]
        if valid_cat_moves:
            cat_action = random.choice(valid_cat_moves)
            self.cat_pos = [self.cat_pos[0] + moves[cat_action][0], self.cat_pos[1] + moves[cat_action][1]]

        reward = -1  

        if self.mouse_pos == self.cat_pos:
            reward -= 100  
            done = True
        elif self.mouse_pos == self.cheese_pos:
            reward += 100  
            done = True
        else:
            done = False

        state = np.array([*self.mouse_pos, *self.cat_pos, *self.cheese_pos], dtype=np.int32)
        return state, reward, done, False, {}

    def render(self):
        pass

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.device = torch.device("cpu")
        print(f"Using device: {self.device}")
        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model(torch.FloatTensor(state).to(self.device))
        return torch.argmax(act_values).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                        torch.max(self.target_model(torch.FloatTensor(next_state).to(self.device))).item())
            target_f = self.model(torch.FloatTensor(state).to(self.device)).detach().clone()
            target_f[action] = target
            self.model.zero_grad()
            loss = self.criterion(self.model(torch.FloatTensor(state).to(self.device)), target_f)
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.update_target_model()  

    def save(self, filepath):
        torch.save(self.model.state_dict(), filepath)

    def load(self, filepath):
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        self.update_target_model()

CELL_SIZE = 50

def init_graphics(grid_size):
    pygame.init()
    window_size = grid_size * CELL_SIZE
    window = pygame.display.set_mode((window_size, window_size))
    pygame.display.set_caption("Cat Mouse Cheese Game")
    clock = pygame.time.Clock()
    return window, clock

def load_images():
    mouse_img = pygame.image.load("img/mouse.png")
    cat_img = pygame.image.load("img/cat.png")
    cheese_img = pygame.image.load("img/cheese.png")
    return mouse_img, cat_img, cheese_img

def draw_grid(window, grid_size):
    window_size = grid_size * CELL_SIZE
    for x in range(0, window_size, CELL_SIZE):
        for y in range(0, window_size, CELL_SIZE):
            rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(window, (200, 200, 200), rect, 1)

def draw_walls(window, walls, grid_size):
    for (i, j), is_wall in walls.items():
        if is_wall:
            x, y = j * CELL_SIZE, i * CELL_SIZE
            rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(window, (0, 0, 0), rect)

def render_entities(window, mouse_pos, cat_pos, cheese_pos, walls, grid_size, mouse_img, cat_img, cheese_img):
    window.fill((255, 255, 255))
    draw_grid(window, grid_size)
    draw_walls(window, walls, grid_size)

    mouse_rect = pygame.Rect(mouse_pos[1] * CELL_SIZE, mouse_pos[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    window.blit(pygame.transform.scale(mouse_img, (CELL_SIZE, CELL_SIZE)), mouse_rect)

    cat_rect = pygame.Rect(cat_pos[1] * CELL_SIZE, cat_pos[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    window.blit(pygame.transform.scale(cat_img, (CELL_SIZE, CELL_SIZE)), cat_rect)

    cheese_rect = pygame.Rect(cheese_pos[1] * CELL_SIZE, cheese_pos[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    window.blit(pygame.transform.scale(cheese_img, (CELL_SIZE, CELL_SIZE)), cheese_rect)

    pygame.display.flip()

def quit_graphics():
    pygame.quit()




def train_dqn(env, agent, episodes=5000, batch_size=64):
    for e in range(episodes):
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32)
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            next_state = np.array(next_state, dtype=np.float32)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        agent.update_target_model()
        print(f"Episodio: {e+1}/{episodes} reward: {reward}")
        
    ############################################################################################################
    agent.save("pth/10x10/dqn_agent_5000.pth")

def test_dqn(env, agent, episodes=10, delay=0.5):
    pygame.init()
    screen, clock = init_graphics(env.grid_size)
    mouse_img, cat_img, cheese_img = load_images()

    for i in range(episodes):
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32)
        done = False
        print(f"\nEpisodio {i+1}/{episodes}\n")
        time.sleep(1)

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                    break

            action = agent.act(state)
            next_state, _, done, _, _ = env.step(action)
            state = np.array(next_state, dtype=np.float32)

            draw_grid(screen, env.grid_size)
            render_entities(screen, env.mouse_pos, env.cat_pos, env.cheese_pos, env.walls, env.grid_size, mouse_img, cat_img, cheese_img)
            pygame.display.flip()
            clock.tick(30)
            time.sleep(delay)

        if state[0:2].tolist() == env.cheese_pos:
            print("Il topo ha raggiunto il formaggio! 🧀")
        else:
            print("Il topo è stato preso dal gatto! 💀")
        time.sleep(1)

    pygame.quit()

def main():
    env = CatMouseCheeseEnv(grid_size=10)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    
    train_dqn(env, agent, episodes=5000)
        
    agent.load("pth/10x10/dqn_agent_1000.pth")
    
    test_dqn(env, agent, episodes=10, delay=0.5)

if __name__ == "__main__":
    main()
    