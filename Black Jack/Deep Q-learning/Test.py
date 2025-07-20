import gymnasium as gym
import numpy as np
from Agent import DQNAgent
from Config import Config
import torch
import os
import time

script_dir = os.path.dirname(os.path.abspath(__file__))

def test():
    # 参数配置
    config = Config()

    # 环境设置
    env = gym.make(config.env_name)
    state_dim = len(env.observation_space.sample())    # 3
    action_dim = env.action_space.n    # 2

    # 创建agent
    agent = DQNAgent(state_dim, action_dim, config.learning_rate, config.gamma,
                    config.buffer_size, config.batch_size, config.target_sync)

    # 加载训练好的模型
    model_path = os.path.join(script_dir, config.model_filename)
    agent.q_net.load_state_dict(torch.load(model_path))

    # 测试
    print('开始测试...')

    # 创建测试环境
    test_env = gym.make(config.env_name, render_mode="human")

    agent.epsilon = 0.0
    final_test_score = []

    for test_episode in range(config.test_episodes):
        test_state, _ = test_env.reset()
        test_reward = 0
        test_done = False

        while not test_done:
            test_action = agent.act(test_state)
            # print(f"选择动作: {test_action}")
            test_state, reward, terminated, truncated, _ = test_env.step(test_action)    # terminated:回合自然结束 truncated:回合被提前终止
            test_done = terminated
            test_reward += reward
            time.sleep(2)

        final_test_score.append(test_reward)
        print(f"Test Episode {test_episode + 1} : Score = {test_reward}")

    print(f"\nAverage Test Score = {np.mean(final_test_score):.2f}")
    print('测试完成\n')

    test_env.close()


if __name__ == '__main__':
    test()
