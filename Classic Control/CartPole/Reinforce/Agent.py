import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PolicyNet(nn.Module):
    def __init__(self, state_action, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_action, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    # 前向传播
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)
    
class PolicyAgent:
    def __init__(self, state_action, action_dim, learning_rate, gamma):
        self.gamma = gamma
        self.policy_net = PolicyNet(state_action, action_dim).to(device)
        
        # 优化器
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)

    def act(self, state):
        state = torch.tensor([state], dtype=torch.float32, device=device)
        probs = self.policy_net(state)    # 计算动作概率
        m = torch.distributions.Categorical(probs)    # 创建分类分布
        action = m.sample()    # 采样动作
        return action.item(), m.log_prob(action)    # 返回动作和对数概率

    def update(self, rewards, log_prob):
        discount = []
        R = 0
        for r in reversed(rewards):   
            R = r + self.gamma * R    # 计算折扣回报
            discount.insert(0, R)
        
        discount = torch.tensor(discount, dtype=torch.float32, device=device)
        discount = (discount - discount.mean()) / (discount.std() + 1e-9)    # 折扣回报标准化
        
        loss = []
        for log_prob, R in zip(log_prob, discount):
            loss.append(-log_prob * R)    # 计算损失

        total_loss = torch.stack(loss).sum()

        self.optimizer.zero_grad()    # 清空梯度
        total_loss.backward()    # 计算损失和梯度
        self.optimizer.step()    # 更新参数

        return total_loss.item()
