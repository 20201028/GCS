"""
一体化配置：训练、验证、测试共享配置
"""
import torch
import os
# def get_best_gpu():
#     """获取负载最低的可用GPU"""
#     if not torch.cuda.is_available():
#         return "cpu"
    
#     import GPUtil
#     gpus = GPUtil.getGPUs()
    
#     if not gpus:
#         return "cpu"
    
#     # 找到使用率最低的GPU
#     best_gpu = min(gpus, key=lambda gpu: gpu.load)
    
#     if best_gpu.load < 0.9:  # 如果GPU使用率低于90%
#         return f"cuda:{best_gpu.id}"
#     else:
#         return "cpu"  # 如果所有GPU都很忙，使用CPU


class Config:
    # === 路径配置 ===
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join("../DataSet/")
    checkpoint_dir = os.path.join(project_root, "checkpoints")
    log_dir = os.path.join(project_root, "logs")
    result_dir = os.path.join(project_root, "results")
    
    # 创建目录
    for dir_path in [checkpoint_dir, log_dir, result_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # === 数据集配置 ===
    dataset_name = "twitter"  # cora, citeseer, facebook, twitter
    # num_nodes = 2708  # 根据数据集调整
    # num_attributes = 1433  # 根据数据集调整
    # 数据划分
    train_ratio = 0.6
    val_ratio = 0.2
    test_ratio = 0.2
    
    # 查询配置
    min_query_nodes = 1
    max_query_nodes = 3
    min_query_attrs = 0
    max_query_attrs = 3
    
    # === 模型架构配置 ===
    # 编码器
    node_feature_dim = 128
    # node_feature_dim = 64
    
    num_gat_heads = 4
    num_encoder_layers = 3
    
    
    # 解码器
    decoder_hidden_dim = 256
    
    # === 训练配置 ===
    batch_size = 1
    num_epochs = 100
    
    patience = 30  # 早停耐心
    n_flows = 1
    # 损失权重
    # beta_query = 1.0   # 查询损失

    # hidden_dim = 256
    # latent_dim = 128
    # dropout_rate = 0.1
    # learning_rate = 1e-3
    # weight_decay = 5e-4
    # beta_kl = 0.01      # KL散度
    # beta_edge = 1    # 边重建
    # beta_constra = 0.5    # 属性与结构一致性
    # temperature = 0.001  # 对比学习温度参数

    #损失加传导
    # learning_rate = 0.002654862640497225
    # weight_decay = 0.0002826552536750886
    # latent_dim= 16
    # hidden_dim= 128
    # dropout_rate= 0.5931272330633232
    # beta_kl= 0.11467939101544004
    # temperature= 0.46958279490714183
    # beta_edge= 0.22446639740571084
    # beta_constra= 0.6238394544939851
    # beta_condu = 0.9212195712863444
    print('最佳超参数：')
    #仅社区损失且公用阈值
    # learning_rate = 0.00036964164892129987
    # weight_decay = 5.570563456184594e-05
    # latent_dim= 128
    # hidden_dim= 128
    # dropout_rate= 0.20242154523584188
    # beta_kl= 0.0
    # temperature= 0.0
    # beta_edge= 0.0
    # beta_constra= 0.0
    # beta_condu = 0.0
    # 仅社区损失、kl且单独阈值
    # learning_rate = 0.004479372415407702
    # weight_decay = 0.0007150412917623105
    # latent_dim= 32
    # hidden_dim= 256
    # dropout_rate= 0.10830499823909731
    # beta_kl= 0.3559620465223241
    # temperature= 0.0
    # beta_edge= 0.0
    # beta_constra= 0.0
    # beta_condu = 0.0
    # 社区、kl、边重建
    # learning_rate = 0.002199922367717476
    # weight_decay = 1.135047960338038e-05
    # latent_dim= 64
    # hidden_dim= 256
    # dropout_rate= 0.5167555958657359
    # beta_kl= 0.17551100330404903
    # temperature= 0.0
    # beta_edge= 0.5690311372404037
    # beta_constra= 0.0
    # beta_condu = 0.0
    # 社区+kl+对比
    # learning_rate = 0.00043606309167878017
    # weight_decay = 0.0007386692981155975
    # latent_dim= 64
    # hidden_dim= 256
    # dropout_rate= 0.3952326568632354
    # beta_kl= 1
    # temperature= 0.23165399358857108
    # beta_edge= 0.0
    # beta_constra= 0.8422028907103616
    # beta_condu = 0.0
    # alpha = 0.2
    # beta = 0.8
    # bce_weight = 50.0
    # kl_annealing = 10.0
    # kl_threshold = 0.2
    # K = 2
    # margin = 2.0
    # beta_query= 0.5

    # learning_rate = 0.0004
    # weight_decay = 0.0007
    # latent_dim= 64
    # hidden_dim= 256
    # dropout_rate= 0.4
    # beta_kl= 0.1
    # temperature= 0.2
    # beta_edge= 0.0
    # beta_constra= 0.8
    # beta_condu = 0.0
    # alpha = 0.2
    # beta = 0.8
    # bce_weight = 50.0
    # kl_annealing = 10.0
    # kl_threshold = 0.2
    # K = 2
    # margin = 2.0
    # beta_query= 0.5

    # learning_rate = 0.001037727966353607
    # weight_decay = 8.473276363620007e-05

    # learning_rate = 0.001
    # weight_decay = 8.5e-05
    # latent_dim= 32
    # hidden_dim= 64
    # dropout_rate= 0.3
    # beta_kl= 0.6
    # temperature= 1.0
    # beta_edge= 0.0
    # beta_constra= 0.1
    # beta_condu = 0.0
    # alpha = 0.4
    # beta = 0.1
    # bce_weight = 80.0
    # kl_annealing = 10.0
    # kl_threshold = 0.6
    # K = 2
    # margin = 1.0
    # beta_query= 0.4
    # beta_supcon = 1  
    # # === 潜空间优化配置 ===
    # latent_lr = 0.1
    # latent_steps = 50
    # latent_reg = 1.0
    # latent_condu = 2.0
    # latent_volume = 0.5
    learning_rate = 0.0005
    # learning_rate = 0.0001
    weight_decay = 4.8e-05
    latent_dim= 128
    hidden_dim= 256
    dropout_rate= 0.4
    beta_kl= 0.8
    temperature= 0.6
    beta_edge= 0.0
    beta_constra= 0.3
    beta_condu = 0.0
    alpha = 0.4
    beta = 0.4
    bce_weight = 80.0
    kl_annealing = 10.0
    kl_threshold = 0.1
    prob_threshold = 0.1
    K = 1
    margin = 5.0
    beta_query= 0.9
    beta_supcon = 0.2  
    # === 潜空间优化配置 ===
    latent_lr = 0.6
    latent_steps = 20
    latent_reg = 1
    latent_condu = 0.3
    latent_volume = 0.5

    # learning_rate = 0.0035068446278660364
    # weight_decay = 0.0008822144887414874
    # latent_dim= 32
    # hidden_dim= 128
    # dropout_rate= 0.6
    # beta_kl= 0.7
    # temperature= 0.8
    # beta_edge= 0.0
    # beta_constra= 1.0
    # beta_condu = 0.0
    # alpha = 1
    # beta = 0.5
    # bce_weight = 80.0
    # kl_annealing = 30.0
    # kl_threshold = 0.7
    # K = 4
    # margin = 2.0
    # beta_query= 0.9
    # learning_rate = 0.00048617291816297517
    # weight_decay = 0.0003639154766485354
    # latent_dim= 32
    # hidden_dim= 64
    # dropout_rate= 0.2
    # beta_kl= 0.7
    # temperature= 0.8
    # beta_edge= 0.0
    # beta_constra= 0.9
    # beta_condu = 0.0
    # alpha = 0.2
    # beta = 1
    # bce_weight = 100.0
    # kl_annealing = 10.0
    # kl_threshold = 0.6
    # K = 2
    # margin = 2.0
    # beta_query= 0.4
    # 社区+kl+对比+边
    # learning_rate = 0.001524343157438817
    # weight_decay = 0.0002629597527146156
    # latent_dim= 64
    # hidden_dim= 128
    # dropout_rate= 0.5927994406361355
    # beta_kl= 0.03335554419065884
    # temperature= 0.40798225263990673
    # beta_edge= 0.9632383119076809
    # beta_constra= 0.6333489064480389
    # beta_condu = 0.0


    # === 潜空间优化配置 ===
    # latent_lr = 0.1
    # latent_momentum = 0.9
    # latent_steps = 50
    # alpha_query_nodes = 2.0
    # beta_query_attrs = 1.5
    # gamma_connectivity = 0.1
    
    # === 推理配置 ===
    threshold = 0.5
    min_community_size = 3
    max_community_size = 50
    # k_hop = 2
    
    # === 设备配置 ===
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    # device = get_best_gpu()
    num_workers = 4
    
    # === 评估配置 ===
    evaluation_metrics = ['f1', 'precision', 'recall', 'jaccard', 'nmi']
    save_results = True
    
    # === 实验配置 ===
    experiment_name = "LSO-CS-FO11111111111"
    random_seed = 42
    debug_mode = False
    
    def __init__(self):
        # 设置随机种子
        # self._set_seed()
        pass
        
    # def _set_seed(self):
    #     """设置随机种子确保可复现性"""
    #     import random
    #     import numpy as np
        
    #     random.seed(self.random_seed)
    #     np.random.seed(self.random_seed)
    #     torch.manual_seed(self.random_seed)
    #     if torch.cuda.is_available():
    #         torch.cuda.manual_seed(self.random_seed)
    #         torch.cuda.manual_seed_all(self.random_seed)
    #         torch.backends.cudnn.deterministic = True
    #         torch.backends.cudnn.benchmark = False

config = Config()