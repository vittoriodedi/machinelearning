import time
import pickle
import random
import pygame
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

class CatMouseCheeseEnv(gym.Env):
    def __init__(self, grid_size=5):
        super(CatMouseCheeseEnv, self).__init__()
        self.grid_size = grid_size
        self.action_space = gym.spaces.Discrete(4)
        self.walls = self.generate_walls()
        self.reset() 
       
    def generate_walls(self):
        walls = {}
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                walls[(i, j)] = {"top": False, "bottom": False, "left": False, "right": False}

        walls[(0, 0)]["right"] = True
        walls[(0, 1)]["left"] = True
        walls[(0, 2)]["bottom"] = True
        walls[(1, 2)]["top"] = True
        walls[(1, 0)]["bottom"] = True
        walls[(2, 0)]["top"] = True
        walls[(1, 1)]["bottom"] = True
        walls[(2, 1)]["top"] = True
        walls[(2, 1)]["right"] = True
        walls[(2, 2)]["left"] = True
        walls[(2, 2)]["right"] = True
        walls[(2, 3)]["left"] = True
        walls[(2, 3)]["top"] = True
        walls[(1, 3)]["bottom"] = True
        walls[(3, 0)]["right"] = True
        walls[(3, 1)]["left"] = True
        walls[(3, 1)]["bottom"] = True
        walls[(4, 1)]["top"] = True
        walls[(3, 3)]["bottom"] = True
        walls[(3, 4)]["bottom"] = True
        walls[(4, 3)]["top"] = True
        walls[(4, 4)]["top"] = True

        return walls
    
    def manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos2[1] - pos2[1])
    
    def can_move(self, pos, action):
        i, j = pos
        if action == 0 and self.walls[(i, j)]["top"]:
            return False
        if action == 1 and self.walls[(i, j)]["bottom"]:
            return False
        if action == 2 and self.walls[(i, j)]["left"]:
            return False
        if action == 3 and self.walls[(i, j)]["right"]:
            return False
        return True
    
    def step(self, action):
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        old_mouse_pos = self.mouse_pos.copy()
        old_cat_pos = self.cat_pos.copy()
        new_mouse_pos = [self.mouse_pos[0] + moves[action][0], self.mouse_pos[1] + moves[action][1]]

        if (0 <= new_mouse_pos[0] < self.grid_size and 0 <= new_mouse_pos[1] < self.grid_size and
                self.can_move(self.mouse_pos, action)):
            self.mouse_pos = new_mouse_pos

        revisit_penalty = -0.5 if tuple(self.mouse_pos) in self.visited_positions else 0
        self.visited_positions.add(tuple(self.mouse_pos))
        idle_penalty = -0.3 if self.mouse_pos == old_mouse_pos else 0
        current_distance = self.manhattan_distance(self.mouse_pos, self.cheese_pos)
        distance_reward = (self.last_distance - current_distance) * 0.5
        self.last_distance = current_distance

        valid_cat_actions = []
        for cat_action in range(4):
            new_cat_pos = [self.cat_pos[0] + moves[cat_action][0], self.cat_pos[1] + moves[cat_action][1]]
            if (0 <= new_cat_pos[0] < self.grid_size and 0 <= new_cat_pos[1] < self.grid_size and
                    self.can_move(self.cat_pos, cat_action)):
                valid_cat_actions.append(cat_action)

        if valid_cat_actions:
            cat_action = random.choice(valid_cat_actions)
            self.cat_pos = [self.cat_pos[0] + moves[cat_action][0], self.cat_pos[1] + moves[cat_action][1]]

        distance_to_cat = self.manhattan_distance(self.mouse_pos, self.cat_pos)
        if distance_to_cat <= 2:
            reward = - (3 - distance_to_cat)

        reward = -0.1 + distance_reward + revisit_penalty + idle_penalty
        done = False
        
        if self.mouse_pos == self.cat_pos or (self.mouse_pos == old_cat_pos and self.cat_pos == old_mouse_pos):
            reward = -30
            done = True
        elif self.mouse_pos == self.cheese_pos:
            reward = 30
            done = True

        next_state = np.array([self.mouse_pos[0], self.mouse_pos[1], self.cheese_pos[0], self.cheese_pos[1], self.cat_pos[0], self.cat_pos[1]], dtype=np.int32)
        return next_state, reward, done, False, {} 
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.mouse_pos = [random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)]
        self.cat_pos = [random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)]
        while self.cat_pos == self.mouse_pos:
            self.cat_pos = [random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)]
        self.cheese_pos = [random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)]
        while self.cat_pos == self.cheese_pos or self.mouse_pos == self.cheese_pos:
            self.cheese_pos = [random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)]
        self.visited_positions = set()
        self.last_distance = self.manhattan_distance(self.mouse_pos, self.cheese_pos)
        state = np.array([self.mouse_pos[0], self.mouse_pos[1], self.cheese_pos[0], self.cheese_pos[1], self.cat_pos[0], self.cat_pos[1]], dtype=np.int32)
        return state, {}
    
 

CELL_SIZE = 100

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
    for (i, j), wall in walls.items():
        x, y = j * CELL_SIZE, i * CELL_SIZE
        if wall["top"]:
            pygame.draw.line(window, (0, 0, 0), (x, y), (x + CELL_SIZE, y), 5)
        if wall["bottom"]:
            pygame.draw.line(window, (0, 0, 0), (x, y + CELL_SIZE), (x + CELL_SIZE, y + CELL_SIZE), 5)
        if wall["left"]:
            pygame.draw.line(window, (0, 0, 0), (x, y), (x, y + CELL_SIZE), 5)
        if wall["right"]:
            pygame.draw.line(window, (0, 0, 0), (x + CELL_SIZE, y), (x + CELL_SIZE, y + CELL_SIZE), 5)

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


def calculate_accuracy(env, q_table, episodes=100):
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

def calculate_rewards(env, q_table, episodes=100):
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

def plot_accuracy(accuracy_per_100_episodes, filename):
    episodes = list(range(100, len(accuracy_per_100_episodes) * 100 + 1, 100))

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, accuracy_per_100_episodes, marker='o')
    plt.xlabel('Numero di Episodi')
    plt.ylabel('Accuratezza (%)')
    plt.title('Accuratezza per 100 Episodi')
    plt.grid(True)
    plt.savefig(filename)
    plt.close() 

def plot_rewards(rewards_per_100_episodes, filename):
    episodes = list(range(100, len(rewards_per_100_episodes) * 100 + 1, 100))

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, rewards_per_100_episodes, marker='o')
    plt.xlabel('Numero di Episodi')
    plt.ylabel('Ricompensa Media')
    plt.title('Ricompensa Media per 100 Episodi')
    plt.grid(True)
    plt.savefig(filename)
    plt.close()  # Chiude la figura senza mostrarla

def train_q_learning(env, episodes=10000, alpha=0.1, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.05):
    q_table = np.zeros((env.grid_size, env.grid_size, env.grid_size, env.grid_size, env.grid_size, env.grid_size, 4))
    accuracy_per_100_episodes = []
    rewards_per_100_episodes = []

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

        if (episode + 1) % 100 == 0:
            accuracy = calculate_accuracy(env, q_table)
            accuracy_per_100_episodes.append(accuracy)

            average_reward = calculate_rewards(env, q_table)
            rewards_per_100_episodes.append(average_reward)

    return q_table, accuracy_per_100_episodes, rewards_per_100_episodes

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
    env = CatMouseCheeseEnv(grid_size=5)
    
    q_table, accuracy_per_100_episodes, rewards_per_100_episodes = train_q_learning(env, episodes=20000)
    save_q_table(q_table, "q_tables/q_table_20000.pkl")
    
    plot_accuracy(accuracy_per_100_episodes, filename="plot/20000/accuracy_per_100_episodes.png")
    plot_rewards(rewards_per_100_episodes, filename="plot/20000/rewards_per_100_episodes.png")
    
    q_table = load_q_table("q_tables/q_table_20000.pkl")
    test_q_learning(env, q_table, episodes=10, delay=0.5)

if __name__ == "__main__":
    main()
