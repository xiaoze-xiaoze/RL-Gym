class Config:
    # 网络参数
    learning_rate = 0.001
    gamma = 0.99
    
    # 经验回放参数
    buffer_size = 20000
    batch_size = 32
    
    # 目标网络同步频率
    target_sync = 100
    
    # 训练参数
    train_episodes = 300
    
    # 测试参数
    test_episodes = 10
    
    # 环境参数
    env_name = 'CartPole-v1'
    max_episode_steps = 2000
    
    # 模型保存路径
    model_filename = 'model/best_model.pth'
