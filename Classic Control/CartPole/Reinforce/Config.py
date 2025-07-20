class Config:
    # 网络参数
    learning_rate = 0.001
    gamma = 0.99
    
    # 训练参数
    train_episodes = 500
    
    # 测试参数
    test_episodes = 10
    
    # 环境参数
    env_name = 'CartPole-v1'
    max_episode_steps = 2000
    
    # 模型保存路径
    model_filename = 'model/best_model.pth'