import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple, deque
import random
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 网络结构
class QNet(nn.Module):
    def __init__(self, state_action, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_action, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    # 前向传播
    def forward(self, x):
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
        states = torch.FloatTensor(batch.state)
        actions = torch.LongTensor(batch.action)
        rewards = torch.FloatTensor(batch.reward)
        next_states = torch.FloatTensor(batch.next_state)
        dones = torch.BoolTensor(batch.done)
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

# Agent
class DQNAgent:
    def __init__(self, state_dim, action_dim, learning_rate, gamma, buffer_size, batch_size, target_sync):
        # 网络
        self.q_net = QNet(state_dim, action_dim).to(device)
        self.target_net = QNet(state_dim, action_dim).to(device)
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
        states, actions, rewards, next_states, dones = map(torch.tensor, self.buffer.sample(self.batch_size))
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