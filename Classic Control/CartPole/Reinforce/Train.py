import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from Agent import PolicyAgent
from Config import Config
import torch
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

def train():
    # 参数配置
    config = Config()

    # 环境设置
    env = gym.make(config.env_name, max_episode_steps=config.max_episode_steps)
    state_dim = env.observation_space.shape[0]    # 4
    action_dim = env.action_space.n    # 2

    # 创建agent
    agent = PolicyAgent(state_dim, action_dim, config.learning_rate, config.gamma)

    # 训练
    score = []
    eval_score = []
    losses = []
    best_score = 0

    print("开始训练...")
    for episode in range(config.train_episodes):
        state, _ = env.reset()

        total_reward = 0
        done = False
        
        # 存储一个episode的数据
        states = []
        actions = []
        rewards = []
        log_probs = []

        while not done:
            action, log_prob = agent.act(state)
            
            next_state, reward, terminated, truncated, _ = env.step(action)    # terminated:回合自然结束 truncated:回合被提前终止
            done = terminated or truncated

            # 存储数据
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)

            state = next_state
            total_reward += reward

        loss = agent.update(rewards, log_probs)
        losses.append(loss)

        print(f"Episode {episode + 1} : Reward = {total_reward}")
        score.append(total_reward)
        
        if (episode + 1) % 10 == 0:
            agent.policy_net.eval()

            current_eval_scores = []
            for eval_episode in range(10): 
                eval_state, _ = env.reset()
                eval_reward = 0
                eval_done = False
                
                while not eval_done:
                    with torch.no_grad():
                        eval_action, _ = agent.act(eval_state)    # 纯贪婪策略选择动作
                    eval_state, reward, terminated, truncated, _ = env.step(eval_action)
                    eval_done = terminated or truncated
                    eval_reward += reward
                
                current_eval_scores.append(eval_reward)    # 记录每个回合的分数

            agent.policy_net.train()

            avg_eval_score = np.mean(current_eval_scores)
            eval_score.append(avg_eval_score)

            print(f"\nEpisode {episode + 1} : Eval Score = {avg_eval_score}\n")

            if avg_eval_score > best_score + 10:
                best_score = avg_eval_score
                model_path = os.path.join(script_dir, config.model_filename)
                # 确保model文件夹存在
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                torch.save(agent.policy_net.state_dict(), model_path)

    print('训练完成\n')

    # 绘制loss曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(score)
    plt.title('Training Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')

    plt.subplot(1, 2, 2)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')

    plt.tight_layout()    # 调整子图布局
    plt.show()


if __name__ == '__main__':
    train()
