import time
import pickle
import random
import pygame
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

class CatMouseCheeseEnv(gym.Env):
    def __init__(self, grid_size=10):
        super(CatMouseCheeseEnv, self).__init__()
        self.grid_size = grid_size
        self.action_space = gym.spaces.Discrete(4)
        self.walls = self.generate_walls()
        self.reset()
        self.last_action = None
       
    def generate_walls(self):
        walls = {(i, j): False for i in range(self.grid_size) for j in range(self.grid_size)}
        static_walls = [(1, 1), (2, 2), (1, 4), (1, 5), (2, 7), (1, 8),
                        (4, 3), (5, 3), (4, 6), (5, 6), (7, 2), (8, 1),
                        (8, 4), (8, 5), (7, 7), (8, 8)]
        for wall in static_walls:
            walls[wall] = True
        return walls
    
    def manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos2[1] - pos2[1])
    
    def can_move(self, pos, action):
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

        old_mouse_pos = self.mouse_pos.copy()
        old_cat_pos = self.cat_pos.copy()

        if self.can_move(self.mouse_pos, action):
            self.mouse_pos = [self.mouse_pos[0] + moves[action][0], self.mouse_pos[1] + moves[action][1]]

        self.last_action = action

        valid_cat_actions = [a for a in moves.keys() if self.can_move(self.cat_pos, a)]
        if valid_cat_actions:
            cat_action = random.choice(valid_cat_actions)
            self.cat_pos = [self.cat_pos[0] + moves[cat_action][0], self.cat_pos[1] + moves[cat_action][1]]

        reward = -0.1

        new_distance_to_cheese = self.manhattan_distance(self.mouse_pos, self.cheese_pos)
        distance_to_cat = self.manhattan_distance(self.mouse_pos, self.cat_pos)

        if new_distance_to_cheese < self.last_distance_to_cheese:
            reward += 10  
        else:
            reward -= 2  
        if distance_to_cat < self.last_distance_to_cat:
            reward -= 5  
        else:
            reward += 2  
            
        self.last_distance_to_cheese = new_distance_to_cheese
        self.last_distance_to_cat = distance_to_cat

        if tuple(self.mouse_pos) in self.visited_positions:
            reward -= 8  
        self.visited_positions.add(tuple(self.mouse_pos))

        if self.mouse_pos == old_mouse_pos:
            reward -= 10  

        done = False
        if self.mouse_pos == self.cat_pos or (self.mouse_pos == old_cat_pos and self.cat_pos == old_mouse_pos):
            reward -= 100  
            done = True
        elif self.mouse_pos == self.cheese_pos:
            reward += 120  
            done = True

        state = np.array([*self.mouse_pos, *self.cat_pos, *self.cheese_pos], dtype=np.int32)
        return state, reward, done, False, {}
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        valid_positions = [(i, j) for i in range(self.grid_size) for j in range(self.grid_size) if not self.walls[(i, j)]]
        positions = random.sample(valid_positions, 3)
        self.mouse_pos, self.cat_pos, self.cheese_pos = map(list, positions)
        self.visited_positions = set()
        self.last_distance_to_cheese = self.manhattan_distance(self.mouse_pos, self.cheese_pos)
        self.last_distance_to_cat = self.manhattan_distance(self.mouse_pos, self.cat_pos)
        state = np.array([*self.mouse_pos, *self.cat_pos, *self.cheese_pos], dtype=np.int32)
        return state, {}
    
 

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

def draw_walls(window, walls):
    for (i, j), is_wall in walls.items():
        if is_wall:
            x, y = j * CELL_SIZE, i * CELL_SIZE
            rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(window, (0, 0, 0), rect)

def render_entities(window, mouse_pos, cat_pos, cheese_pos, walls, grid_size, mouse_img, cat_img, cheese_img):
    window.fill((255, 255, 255))  
    draw_grid(window, grid_size)
    draw_walls(window, walls)

    mouse_rect = pygame.Rect(mouse_pos[1] * CELL_SIZE, mouse_pos[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    window.blit(pygame.transform.scale(mouse_img, (CELL_SIZE, CELL_SIZE)), mouse_rect)

    cat_rect = pygame.Rect(cat_pos[1] * CELL_SIZE, cat_pos[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    window.blit(pygame.transform.scale(cat_img, (CELL_SIZE, CELL_SIZE)), cat_rect)

    cheese_rect = pygame.Rect(cheese_pos[1] * CELL_SIZE, cheese_pos[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    window.blit(pygame.transform.scale(cheese_img, (CELL_SIZE, CELL_SIZE)), cheese_rect)

    pygame.display.flip()  

def quit_graphics():
    pygame.quit()


def calculate_accuracy(env, q_table, episodes=1000):
    successes = 0

    for _ in range(episodes):
        state, _ = env.reset()
        done = False

        while not done:
            action = np.argmax(q_table[state[0], state[1], state[2], state[3], state[4], state[5]])
            state, _, done, _, _ = env.step(action)

        if state[0:2].tolist() == env.cheese_pos:
            successes += 1

    accuracy = (successes / episodes) * 100
    return accuracy

def calculate_rewards(env, q_table, episodes=1000):
    total_rewards = 0

    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = np.argmax(q_table[state[0], state[1], state[2], state[3], state[4], state[5]])
            state, reward, done, _, _ = env.step(action)
            episode_reward += reward

        total_rewards += episode_reward

    average_reward = total_rewards / episodes
    return average_reward

def plot_accuracy(accuracy_per_1000_episodes, filename):
    episodes = list(range(1000, len(accuracy_per_1000_episodes) * 1000 + 1, 1000))
    
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, accuracy_per_1000_episodes, marker='o')    
    plt.xlabel('Numero di Episodi')
    plt.ylabel('Accuratezza (%)')
    plt.title('Accuratezza per 1000 Episodi')
    plt.grid(True)
    plt.savefig(filename)
    plt.close() 

def plot_rewards(rewards_per_1000_episodes, filename):
    episodes = list(range(1000, len(rewards_per_1000_episodes) * 1000 + 1, 1000))

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, rewards_per_1000_episodes, marker='o')
    plt.xlabel('Numero di Episodi')
    plt.ylabel('Ricompensa Media')
    plt.title('Ricompensa Media per 1000 Episodi')
    plt.grid(True)
    plt.savefig(filename)
    plt.close()  

def train_q_learning(env, episodes=10000, alpha=0.1, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.05):
    q_table = np.zeros((env.grid_size, env.grid_size, env.grid_size, env.grid_size, env.grid_size, env.grid_size, 4))
    accuracy_per_1000_episodes = []
    rewards_per_1000_episodes = []

    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state[0], state[1], state[2], state[3], state[4], state[5]])

            next_state, reward, done, _, _ = env.step(action)

            q_table[state[0], state[1], state[2], state[3], state[4], state[5], action] = (1 - alpha) * q_table[state[0], state[1], state[2], state[3], state[4], state[5], action] + \
                alpha * (reward + gamma * np.max(q_table[next_state[0], next_state[1], next_state[2], next_state[3], next_state[4], next_state[5]]))

            state = next_state
            total_reward += reward

        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        if (episode + 1) % 1000 == 0:
            accuracy = calculate_accuracy(env, q_table)
            accuracy_per_1000_episodes.append(accuracy)

            average_reward = calculate_rewards(env, q_table)
            rewards_per_1000_episodes.append(average_reward)

    return q_table, accuracy_per_1000_episodes, rewards_per_1000_episodes

def test_q_learning(env, q_table, episodes=10, delay=0.5):
    pygame.init()
    screen, clock = init_graphics(env.grid_size)
    mouse_img, cat_img, cheese_img = load_images()

    for i in range(episodes):
        state, _ = env.reset()
        done = False
        print(f"\n Episodio {i+1}/{episodes} \n")
        time.sleep(1)

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                    break

            action = np.argmax(q_table[state[0], state[1], state[2], state[3], state[4], state[5]])
            state, _, done, _, _ = env.step(action)

            draw_grid(screen, env.grid_size)
            render_entities(screen, env.mouse_pos, env.cat_pos, env.cheese_pos, env.walls, env.grid_size, mouse_img, cat_img, cheese_img)
            pygame.display.flip()
            clock.tick(30)
            time.sleep(delay)

        if state[0:2].tolist() == env.cheese_pos:
            print("Il topo ha raggiunto il formaggio! 🧀 ")
        else:
            print("Il topo è stato preso dal gatto! 💀")
        time.sleep(1)

    pygame.quit()
    
def save_q_table(q_table, filename="q_table.pkl"):
    with open(filename, "wb") as file:
        pickle.dump(q_table, file)
        
def load_q_table(filename="q_table.pkl"):
    with open(filename, "rb") as file:
        return pickle.load(file)

def main():
    env = CatMouseCheeseEnv(grid_size=10)
    
    #q_table, accuracy_per_1000_episodes, rewards_per_1000_episodes = train_q_learning(env, episodes=10000)
    #save_q_table(q_table, "q_tables/q_table_10000.pkl")
    
    #plot_accuracy(accuracy_per_1000_episodes, filename="plot/10000/accuracy_per_1000_episodes.png")
    #plot_rewards(rewards_per_1000_episodes, filename="plot/10000/rewards_per_1000_episodes.png")
    
    q_table = load_q_table("q_tables/q_table_100000.pkl")
    test_q_learning(env, q_table, episodes=10, delay=0.5)

if __name__ == "__main__":
    main()
