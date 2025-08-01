import gymnasium as gym
from Agent import SACAgent
import os

class Config:
    # 网络参数
    learning_rate = 0.001
    gamma = 0.99
    tau = 0.005    # 软更新参数
    
    # 经验回放参数
    buffer_size = 100000
    batch_size = 256
    
    # 训练参数
    train_episodes = 500
    
    # 测试参数
    test_episodes = 10
    
    # 环境参数
    env_name = 'Pendulum-v1'
    max_episode_steps = 200
    
    # 模型保存路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_filename = os.path.join(script_dir, 'model', 'best_model.pth')

def create_agent():
    config = Config()
    
    # 创建环境
    env = gym.make(config.env_name, max_episode_steps=config.max_episode_steps)
    state_dim = env.observation_space.shape[0]  # 3
    action_dim = env.action_space.shape[0]  # 1
    max_action = float(env.action_space.high[0])  # 2.0
    
    # 创建Agent
    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        buffer_size=config.buffer_size,
        batch_size=config.batch_size,
        tau=config.tau
    )
    
    return agent, env, config

def train():
    agent, env, config = create_agent()
    
    # 训练
    train_scores, eval_scores, losses = agent.train(env, config)
    
    env.close()
    
    return train_scores, eval_scores, losses

def test():
    agent, _, config = create_agent()
    
    # 加载训练好的模型
    agent.load_model(config.model_filename)

    # 创建带渲染的测试环境
    test_env = gym.make(config.env_name, render_mode="human", max_episode_steps=None)
    
    # 测试
    test_scores, avg_score = agent.test(test_env, episodes=config.test_episodes)
    
    test_env.close()

    return test_scores, avg_score

if __name__ == '__main__':
    train()
    test()
