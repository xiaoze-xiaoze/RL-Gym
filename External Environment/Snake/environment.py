import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random

class Config:
    GridSize = 13
    WindowSize = 480

    RewardFood = 10.0
    RewardDeath = -10.0
    RewardStep = -0.1

    InitialSnakeLength = 3
    MaxSteps = 1200
    FPS = 30

class Snake:
    def __init__(self, config):
        self.config = config

        self.directions = {
            0: (0, -1),    # up
            1: (1, 0),     # right
            2: (0, 1),     # down
            3: (-1, 0)     # left
        }
        
        self.reset()

    def reset(self):
        center = self.config.GridSize // 2
        self.snake = []
        for i in range(self.config.InitialSnakeLength):
            self.snake.append((center, center - i))

        self.direction = (0, 1)
        self.score = 0
        self.steps = 0
        self.done = False
        self.terminated = False  
        self.truncated = False  

        self.foods()

    def foods(self):
        while True:
            food = (random.randint(0, self.config.GridSize - 1), random.randint(0, self.config.GridSize - 1))
            if food not in self.snake:
                self.food = food
                break
    
    def distance(self):
        if not self.snake or not self.food:
            return 0
        head = self.snake[0]
        return abs(head[0] - self.food[0]) + abs(head[1] - self.food[1])
    
    def update(self, action):
        new_direction = self.directions[action]
        # 检查是否为反向移动
        if (new_direction[0] * -1, new_direction[1] * -1) != self.direction:
            self.direction = new_direction

    def collision(self, pos):
        x, y = pos

        if (x < 0 or x >= self.config.GridSize or 
            y < 0 or y >= self.config.GridSize):
            return True
        
        if pos in self.snake:
            return True
        
        return False

    def move(self, action):
        if self.done:
            return 
        
        self.steps += 1

        self.update(action)

        head_x, head_y = self.snake[0]
        new_head = (head_x + self.direction[0], head_y + self.direction[1])

        if self.collision(new_head):
            self.done = True
            self.terminated = True  # 撞墙或撞自己
            self.truncated = False
            return

        self.snake.insert(0, new_head)

        if new_head == self.food:
            self.score += 1
            self.foods()
        else:
            self.snake.pop()

        if self.steps >= self.config.MaxSteps:
            self.done = True
            self.terminated = False    # 超时不算正常结束
            self.truncated = True      # 超时算截断

    def get_observation(self):
        size = self.config.GridSize
        grid = np.zeros((size, size, 4), dtype=np.float32)
        
        # 蛇头
        if self.snake:
            head_x, head_y = self.snake[0]
            grid[head_y, head_x, 0] = 1.0
        
        # 蛇身
        for x, y in self.snake[1:]:
            grid[y, x, 1] = 1.0
        
        # 食物
        if self.food:
            food_x, food_y = self.food
            grid[food_y, food_x, 2] = 1.0
        
        # 边界
        grid[0, :, 3] = 1.0             
        grid[size-1, :, 3] = 1.0         
        grid[:, 0, 3] = 1.0              
        grid[:, size-1, 3] = 1.0       
        
        return grid

class Environment(gym.Env):
    def __init__(self, config):
        self.config = config
        self.game = Snake(config)

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low = 0,
            high = 1,
            shape = (self.config.GridSize, self.config.GridSize, 4),
            dtype = np.float32
        )

        self.screen = None
        self.clock = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset()
        return self.game.get_observation(), {}

    def step(self, action):
        old_score = self.game.score

        self.game.move(action)

        # 计算奖励
        if self.game.terminated:
            reward = self.config.RewardDeath
        elif self.game.truncated:
            reward = -5.0
        elif self.game.score > old_score:
            reward = self.config.RewardFood
        else:
            reward = self.config.RewardStep

        obs = self.game.get_observation()
        
        info = {
            'score': self.game.score,
            'length': len(self.game.snake),
            'steps': self.game.steps
        }

        return obs, reward, self.game.terminated, self.game.truncated, info

    def render(self):
        cell_size = self.config.WindowSize // self.config.GridSize

        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.config.WindowSize, self.config.WindowSize))
            pygame.display.set_caption("Snake")
            self.clock = pygame.time.Clock()

         # 处理 pygame 事件，防止卡死
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        # 填充背景
        self.screen.fill((0, 0, 0))

        # 绘制蛇 - 分数影响渐变
        snake_length = len(self.game.snake)
        if snake_length > 0:  # 防止除零错误
            score_factor = min(self.game.score / 10.0, 1.0)    # 分数影响因子
            
            for i, (x, y) in enumerate(self.game.snake):
                if i == 0:
                    color = (255, 255, 0)  # 蛇头保持黄色
                else:
                    # 分数越高，渐变越明显
                    base_green = 255
                    fade_amount = int((i / snake_length) * 200 * score_factor)
                    green = max(base_green - fade_amount, 55)
                    red = int(fade_amount * 0.5)
                    color = (red, green, 128)
                
                rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
                pygame.draw.rect(self.screen, color, rect)


        # 绘制食物
        if hasattr(self.game, 'food') and self.game.food:
            food_x, food_y = self.game.food
            food_rect = pygame.Rect(food_x * cell_size, food_y * cell_size, cell_size, cell_size)
            pygame.draw.rect(self.screen, (255, 0, 0), food_rect)

        # 更新屏幕
        pygame.display.flip()
        self.clock.tick(self.config.FPS)

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None