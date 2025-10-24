import gymnasium as gym
from Agent import DQNAgent
import sys
import os

# 添加上级目录到路径，以便导入environment模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment import Environment, Config as EnvConfig

class Config:
    # 网络参数
    learning_rate = 0.001
    gamma = 0.99
    
    # 经验回放参数
    buffer_size = 200000
    batch_size = 64
    
    # 目标网络同步频率
    target_sync = 500
    
    # 训练参数
    train_episodes = 20000
    
    # 测试参数
    test_episodes = 10
    
    # 模型保存路径 - 使用脚本所在目录的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_filename = os.path.join(script_dir, 'model', 'best_model.pth')

def create_agent():
    config = Config()
    
    env_config = EnvConfig()
    env = Environment(env_config)
    
    # 获取环境参数
    input_channels = 4    # 4个通道
    grid_size = env_config.GridSize    # 13
    action_dim = env.action_space.n    # 4
    
    # 创建Agent
    agent = DQNAgent(
        input_channels=input_channels,
        grid_size=grid_size,
        action_dim=action_dim,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        buffer_size=config.buffer_size,
        batch_size=config.batch_size,
        target_sync=config.target_sync
    )
    
    return agent, env, config

def train():
    agent, env, config = create_agent()
    
    # 训练
    train_scores, eval_scores, losses, steps_per_episode, game_scores = agent.train(env, config)
    
    env.close()
    
    return train_scores, eval_scores, losses, steps_per_episode, game_scores

def test():
    agent, _, config = create_agent()
    
    # 加载训练好的模型
    agent.load_model(config.model_filename)

    env_config = EnvConfig()
    test_env = Environment(env_config)
    
    # 测试
    test_scores, avg_score, test_steps, avg_steps = agent.test(test_env, episodes=config.test_episodes, render=True)
    
    test_env.close()

    return test_scores, avg_score, test_steps, avg_steps

if __name__ == '__main__':
    # train()
    test()