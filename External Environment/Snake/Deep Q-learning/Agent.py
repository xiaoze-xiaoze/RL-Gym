import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple, deque
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 网络结构
class QNet(nn.Module):
    def __init__(self, input_channels, grid_size, action_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        conv_output_size = 4 * 4 * 32
        
        self.fc1 = nn.Linear(conv_output_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_dim)

    # 前向传播
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = x.reshape(x.size(0), -1)    # 展平

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
# 经验回放
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)    # 经验回放缓冲区

    # 交互push进来
    def push(self, *args):
        self.buffer.append(Transition(*args))

    # 随机采集一个batch
    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)    # 随机采样
        batch = Transition(*zip(*transitions))    # 拆包

        states = np.array(batch.state)
        next_states = np.array(batch.next_state)
        
        if len(states.shape) == 4:    # (batch, height, width, channels)
            states = np.transpose(states, (0, 3, 1, 2))
            next_states = np.transpose(next_states, (0, 3, 1, 2))

        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(batch.action).to(device)
        rewards = torch.FloatTensor(batch.reward).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.BoolTensor(batch.done).to(device)

        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

# Agent
class DQNAgent:
    def __init__(self, input_channels, grid_size, action_dim, learning_rate, gamma, buffer_size, batch_size, target_sync):
        # 网络
        self.q_net = QNet(input_channels, grid_size, action_dim).to(device)
        self.target_net = QNet(input_channels, grid_size, action_dim).to(device)
        # 同步主网络和影子网络
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()    # 影子网络只推理不求梯度

        # 优化器
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=learning_rate)

        # 经验池
        self.buffer = ReplayBuffer(buffer_size)

        # 探索和训练计数
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.step_count = 0

        # 超参数
        self.action_dim = action_dim    
        self.gamma = gamma    # 折扣因子
        self.batch_size = batch_size
        self.target_sync = target_sync    # 目标网络同步频率


    def act(self, state):
        # ε-贪婪策略
        if random.random() < self.epsilon:
            return np.random.randint(self.action_dim)

        if len(state.shape) == 3:    # (height, width, channels)
            state = np.transpose(state, (2, 0, 1))    # 转换为 (channels, height, width)
        
        # 贪婪动作
        state = torch.FloatTensor(state).unsqueeze(0).to(device)    # unsqueeze(0)增加一个维度
        with torch.no_grad():
            q_values = self.q_net(state)    # 计算Q值
        return q_values.argmax().item()    # 返回动作索引
        
    def remember(self, *args):
        self.buffer.push(*args)

    def update(self):
        # 经验回放
        if len(self.buffer) < self.batch_size:
            return
        
        # 采样经验
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states = states.float().to(device)
        actions = actions.long().to(device)
        rewards = rewards.float().to(device)
        next_states = next_states.float().to(device)
        dones = dones.bool().to(device)

        # 当前Q值
        current_q = self.q_net(states).gather(1, actions.unsqueeze(1))

        # 目标Q值
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (self.gamma * next_q * (~dones))    # ~dones:布尔取反

        # 计算损失和更新
        loss = F.mse_loss(current_q.squeeze(), target_q)
        
        # 更新网络
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新ε
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # 同步目标网络
        self.step_count += 1
        if self.step_count % self.target_sync == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

    def train(self, env, config):
        print("开始训练...")
        
        # 训练记录
        score = []
        game_scores = []  # 添加游戏分数统计（吃了多少食物）
        eval_score = []
        losses = []
        steps_per_episode = []  # 添加步数统计
        best_score = -float('inf')
        
        for episode in range(config.train_episodes):
            state, _ = env.reset()
            total_reward = 0
            episode_steps = 0  # 当前episode的步数
            done = False
            episode_losses = []

            while not done:
                action = self.act(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                self.remember(state, action, reward, next_state, done)
                loss = self.update()
                if loss is not None:
                    episode_losses.append(loss)

                state = next_state
                total_reward += reward
                episode_steps += 1  # 步数计数

            print(f"Episode {episode + 1} : Reward = {total_reward:.1f}  Steps = {episode_steps}  Score = {info['score']}")
            losses.append(np.mean(episode_losses) if episode_losses else 0)
            score.append(total_reward)
            game_scores.append(info['score'])  # 记录游戏分数
            steps_per_episode.append(episode_steps)  # 记录步数
            
            # 每10个episode评估一次
            if (episode + 1) % 30 == 0:
                avg_eval_score = self.evaluate(env, episodes=10)
                eval_score.append(avg_eval_score)
                avg_steps = np.mean(steps_per_episode[-10:])  # 最近10个episode的平均步数
                avg_game_score = np.mean(game_scores[-10:])  # 最近10个episode的平均游戏分数
                print(f"\nEpisode {episode + 1} : Eval Reward = {avg_eval_score:.1f}  Avg Steps = {avg_steps:.1f}  Avg Score = {avg_game_score:.1f}\n")


                # 保存最佳模型
                if avg_eval_score > best_score:
                    best_score = avg_eval_score
                    self.save_model(config.model_filename)
                    print('保存模型\n')

        print('训练完成\n')
        
        # 绘制训练曲线
        self.plot(score, losses, steps_per_episode)  # 传入步数数据
        
        return score, eval_score, losses, steps_per_episode, game_scores  # 返回步数数据和游戏分数

    def evaluate(self, env, episodes=10):
        # 保存当前epsilon和训练状态
        eval_epsilon = self.epsilon
        self.epsilon = 0.0  # 纯贪婪策略
        self.q_net.eval()
        
        eval_scores = []
        for eval_episode in range(episodes): 
            eval_state, _ = env.reset()
            eval_reward = 0
            eval_done = False
            
            while not eval_done:
                with torch.no_grad():
                    eval_action = self.act(eval_state)
                eval_state, reward, terminated, truncated, _ = env.step(eval_action)
                eval_done = terminated or truncated
                eval_reward += reward
            
            eval_scores.append(eval_reward)
        
        # 恢复训练状态
        self.epsilon = eval_epsilon
        self.q_net.train()
        
        return np.mean(eval_scores)

    def save_model(self, model_path):
        # 确保目录存在
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(self.q_net.state_dict(), model_path)

    def load_model(self, model_path):
        self.q_net.load_state_dict(torch.load(model_path))

    def test(self, env, episodes=10, render=True):
        print('开始测试...')
        
        # 设置为纯贪婪策略
        self.epsilon = 0.0
        self.q_net.eval()
        
        test_scores = []
        test_steps = []  # 添加步数统计
        
        for test_episode in range(episodes):
            test_state, _ = env.reset()
            test_reward = 0
            episode_steps = 0  # 当前episode的步数
            test_done = False

            while not test_done:
                test_action = self.act(test_state)
                test_state, reward, terminated, truncated, info = env.step(test_action)
                test_done = terminated or truncated
                test_reward += reward
                episode_steps += 1  # 步数计数

                if render:
                    env.render()

            test_scores.append(test_reward)
            test_steps.append(episode_steps)  # 记录步数
            print(f"Test Episode {test_episode + 1} : Score = {info['score']}  Reward = {test_reward:.1f}  Steps = {episode_steps}")
            time.sleep(1)

        avg_score = np.mean(test_scores)
        avg_steps = np.mean(test_steps)  # 平均步数
        print(f"\nAverage Test Score = {avg_score}")
        print(f"Average Test Steps = {avg_steps:.1f}")
        print('测试完成\n')
        
        return test_scores, avg_score, test_steps, avg_steps  # 返回步数数据

    def plot(self, scores, losses, steps=None):
        if steps is not None:
            plt.figure(figsize=(18, 5))
            
            plt.subplot(1, 3, 1)
            plt.plot(scores)
            plt.title('Training Reward')
            plt.xlabel('Episode')
            plt.ylabel('Reward')

            plt.subplot(1, 3, 2)
            plt.plot(losses)
            plt.title('Training Loss')
            plt.xlabel('Episode')
            plt.ylabel('Loss')
            
            plt.subplot(1, 3, 3)
            plt.plot(steps)
            plt.title('Steps per Episode')
            plt.xlabel('Episode')
            plt.ylabel('Steps')

        plt.tight_layout()
        plt.show()