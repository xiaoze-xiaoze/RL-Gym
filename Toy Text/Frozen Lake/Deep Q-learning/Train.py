import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from Agent import DQNAgent
from Config import Config
import torch
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

def train():
    # 参数配置
    config = Config()

    # 环境设置
    env = gym.make(config.env_name, max_episode_steps=config.max_episode_steps, is_slippery=config.is_slippery)
    state_dim = env.observation_space.n    # 16
    action_dim = env.action_space.n    # 4

    # 创建agent
    agent = DQNAgent(state_dim, action_dim, config.learning_rate, config.gamma,
                    config.buffer_size, config.batch_size, config.target_sync)

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
        episode_losses = []

        while not done:
            action = agent.act(state)
            
            next_state, reward, terminated, truncated, _ = env.step(action)    # terminated:回合自然结束 truncated:回合被提前终止
            done = terminated or truncated

            agent.remember(state, action, reward, next_state, done)
            loss = agent.update()    # 获取损失值
            if loss is not None:
                episode_losses.append(loss)

            state = next_state
            total_reward += reward

        print(f"Episode {episode + 1} : Reward = {total_reward}")

        losses.append(np.mean(episode_losses))

        score.append(total_reward)
        
        if (episode + 1) % 10 == 0:
            eval_epsilon = agent.epsilon
            agent.epsilon = 0.0
            agent.q_net.eval()
            
            current_eval_scores = []
            for eval_episode in range(20): 
                eval_state, _ = env.reset()
                eval_reward = 0
                eval_done = False
                
                while not eval_done:
                    with torch.no_grad():
                        eval_action = agent.act(eval_state)    # 纯贪婪策略选择动作
                    eval_state, reward, terminated, truncated, _ = env.step(eval_action)
                    eval_done = terminated or truncated
                    eval_reward += reward
                
                current_eval_scores.append(eval_reward)    # 记录每个回合的分数
            
            # 恢复训练时的epsilon
            agent.epsilon = eval_epsilon
            agent.q_net.train()

            avg_eval_score = np.mean(current_eval_scores)
            eval_score.append(avg_eval_score)

            print(f"\nEpisode {episode + 1} : Eval Score = {avg_eval_score}\n")

            if avg_eval_score > best_score + 0.05:
                best_score = avg_eval_score
                model_path = os.path.join(script_dir, config.model_filename)
                # 确保model文件夹存在
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                torch.save(agent.q_net.state_dict(), model_path)

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