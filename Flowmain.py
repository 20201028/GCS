"""
主入口：训练、验证、测试一体化（基于GCN风格改写）
"""
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import json
import os
import sys
import time
from datetime import datetime
from tqdm import tqdm
import warnings
from sklearn.metrics import f1_score, precision_score, recall_score
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config
from sampleQuery import CommunityQueryLoader
from models.Flow_lso_model import LSOCommunitySearch
from utils.metrics import CommunityMetrics
from utils.logger import Logger
from collections import OrderedDict


def post_process_connectivity(probs, threshold, edge_index, query_nodes):
    import networkx as nx
    # 1. 基于阈值初步筛选
    preds = (probs > threshold).astype(int)
    selected_nodes = np.where(preds == 1)[0]
    
    if len(selected_nodes) == 0:
        return preds
    
    # 2. 构建子图，寻找包含 Query 的连通分量
    
    if torch.is_tensor(edge_index):
        edge_index = edge_index.cpu().numpy()
    G_full = nx.Graph()
    for i in range(edge_index.shape[1]):
        u, v = edge_index[0, i], edge_index[1, i]
        G_full.add_edge(u, v)
    
    # 提取导出子图
    G_sub = G_full.subgraph(selected_nodes)

    components = list(nx.connected_components(G_sub))
    refined_preds = np.zeros_like(preds)
    visited = set()
    for q in query_nodes:
        for i, comp in enumerate(components):
            if q in comp:
                if i not in visited:
                    refined_preds[list(comp)] = 1
                    visited.add(i)
                break

    return refined_preds

class Trainer:
    """一体化训练器：基于GCN风格的训练策略"""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device(config.device) if isinstance(config.device, torch.device) else torch.device(config.device)
        self.model.to(self.device)
        
        # 优化器（与原项目相同）
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # 不再使用ReduceLROnPlateau，改为在训练循环中调整
        self.lr_scheduler = None
        
        # 最佳阈值（在验证集上搜索得到）
        self.best_threshold = 0.5
        
        # 记录最佳性能
        self.best_val_f1 = -1
        self.best_test_f1 = -1
        self.best_val_metrics = {}
        self.best_test_metrics = {}
        self.all_predictions = []
        
        # 早停
        self.patience = config.patience
        self.best_val_loss = float('inf')
        self.early_stop_counter = 0
        
        # 记录器
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger = Logger(
            log_dir=os.path.join(config.log_dir, f"{config.experiment_name}_{timestamp}"),
            config=config
        )
        
        # 评估器
        # self.metrics = CommunityMetrics()
        
    def adjust_lr_poly(self, epoch, tot_epoch, power=0.9):
        """多项式学习率衰减（与原项目相同）"""
        lr = self.config.learning_rate * (1 - epoch * 1. / tot_epoch) ** power
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr
    
    def train_epoch(self, train_loader, epoch, use_query_attr):
        """训练一个epoch（与原项目类似）"""
        self.model.train()
        epoch_loss = 0
        epoch_metrics = {}
        
        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            # 移动到设备
            batch = self._move_to_device(batch)
            
            # 前向传播
            outputs = self.model(batch, mode='train',attr=use_query_attr)
            
            # 计算损失
            loss, loss_dict = self.model.compute_loss(outputs, batch, epoch)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # 记录损失
            epoch_loss += loss.item()
            for key, value in loss_dict.items():
                if key not in epoch_metrics:
                    epoch_metrics[key] = 0
                epoch_metrics[key] += value
            
            # 更新进度条
            pbar.set_postfix({
                'loss': loss.item(),
                'comm': loss_dict.get('comm_loss', 0)
            })
            
            # 记录到日志
            if batch_idx % 10 == 0:
                self.logger.log_training_step(
                    epoch, batch_idx, loss.item(), loss_dict
                )
        
        # 计算平均
        avg_loss = epoch_loss / len(train_loader)
        for key in epoch_metrics:
            epoch_metrics[key] /= len(train_loader)
        
        return avg_loss, epoch_metrics
    
    def evaluate_on_val(self, val_loader, epoch, use_query_attr):
        """验证集评估（搜索最佳阈值）"""
        self.model.eval()
        all_outputs = []
        all_labels = []
        all_queries = []
        all_edge_indexes = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch}"):
                batch = self._move_to_device(batch)
                
                # 获取模型输出（概率）
                outputs = self.model(batch, mode='val', attr = use_query_attr)
                probs = torch.sigmoid(outputs['community_pred']).cpu().numpy()
                
                # 获取真实标签
                labels = batch['target']['community_mask'].cpu().numpy()
                # 获取查询节点
                queries = batch['query']['nodes'].cpu().numpy()
                # queries = None
                
                all_outputs.append(probs)
                all_labels.append(labels)
                all_queries.append(queries)
                all_edge_indexes.append(batch['graph']['edge_index'])
        
        # 搜索最佳阈值
        best_threshold, best_f1 = self._search_best_threshold(all_outputs, all_labels)
        
        # 使用最佳阈值计算验证集指标
        val_metrics, all_predictions = self._compute_metrics_with_threshold(
            all_outputs, all_labels, all_queries, best_threshold, all_edge_indexes, test = False
        )
        
        return best_threshold, best_f1, val_metrics
    
    def test_with_best_threshold(self, test_loader, threshold, use_query_attr):
        """使用最佳阈值进行测试"""
        self.model.eval()
        
        all_outputs = []
        all_labels = []
        all_queries = []
        all_edge_indexes = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                batch = self._move_to_device(batch)
                
                # 获取模型输出
                outputs = self.model(batch, mode='test', attr = use_query_attr)
                
                probs = torch.sigmoid(outputs['community_pred']).cpu().numpy()
                print(f'max probs: {probs.max()},min probs: {probs.min()}')
                # 获取真实标签
                labels = batch['target']['community_mask'].cpu().numpy()
                # 获取查询节点
                queries = batch['query']['nodes'].cpu().numpy()
                # queries = None
                edge_index = batch['graph']['edge_index'].cpu().numpy()
                all_edge_indexes.append(edge_index)
                all_outputs.append(probs)
                all_labels.append(labels)
                all_queries.append(queries)
        
        # 使用给定阈值计算测试集指标
        test_metrics, all_predictions = self._compute_metrics_with_threshold(
            all_outputs, all_labels, all_queries, threshold, all_edge_indexes, test = False
        )
        
        return test_metrics, all_predictions
    
    def _search_best_threshold(self, all_outputs, all_labels):
        """搜索最佳阈值（0.05-0.95范围内）"""
        best_threshold = 0.5
        best_f1 = 0
        
        thresholds = np.arange(0.05, 0.96, 0.01)
        
        for thr in thresholds:
            total_f1 = 0
            count = 0
            
            for probs, labels in zip(all_outputs, all_labels):
                # 应用阈值
                predictions = (probs > thr).astype(int)
                # print(predictions)
                # 计算F1分数
                if len(labels) > 0 and len(predictions) > 0:
                    f1 = f1_score(labels, predictions)
                    total_f1 += f1
                    count += 1
            
            if count > 0:
                avg_f1 = total_f1 / count
                if avg_f1 > best_f1:
                    best_f1 = avg_f1
                    best_threshold = thr
        
        self.logger.info(f"Best threshold found: {best_threshold:.3f} (F1: {best_f1:.4f})")
        return best_threshold, best_f1
    
    def _compute_metrics_with_threshold(self, all_outputs, all_labels, all_queries, threshold, all_edge_indexes, test):
        """使用给定阈值计算各种指标"""
        metrics = {
            'f1': [],
            'precision': [],
            'recall': [],
            'jaccard': [],
            'nmi': [],
            'ari': []
        }
        all_predictions = []
        for probs, labels, queries, edge_index in zip(all_outputs, all_labels, all_queries, all_edge_indexes):
            # 应用阈值
            if test is False:
                predictions = (probs > threshold).astype(int)
            # predictions = (probs > threshold).astype(int)
            else:
                predictions = post_process_connectivity(probs, threshold, edge_index, queries)
            all_predictions.append(predictions)
            # 计算各项指标
            if len(labels) > 0 and len(predictions) > 0:
                metrics['f1'].append(f1_score(labels, predictions))
                metrics['precision'].append(precision_score(labels, predictions))
                metrics['recall'].append(recall_score(labels, predictions))

        
        # 计算平均和标准差
        avg_metrics = {}
        for metric_name, scores in metrics.items():
            if scores:
                avg_metrics[f'avg_{metric_name}'] = np.mean(scores)
                avg_metrics[f'std_{metric_name}'] = np.std(scores)
            else:
                avg_metrics[f'avg_{metric_name}'] = 0
                avg_metrics[f'std_{metric_name}'] = 0
        
        return avg_metrics, all_predictions
    
    def _move_to_device(self, batch):
        """移动数据到设备"""
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, dict):
                device_batch[key] = self._move_to_device(value)
            elif isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            else:
                device_batch[key] = value
        return device_batch
    
    # def _save_checkpoint(self, epoch, metrics, name):
    #     """保存检查点"""
    #     checkpoint = {
    #         'epoch': epoch,
    #         'model_state_dict': self.model.state_dict(),
    #         'optimizer_state_dict': self.optimizer.state_dict(),
    #         'best_threshold': self.best_threshold,
    #         'best_val_f1': self.best_val_f1,
    #         'best_val_metrics': self.best_val_metrics,
    #         # 'best_test_metrics': self.best_test_metrics,
    #         'metrics': metrics,
    #         'config': self.config
    #     }
        
    #     path = os.path.join(
    #         self.config.checkpoint_dir,
    #         f"{self.config.experiment_name}_{name}.pth"
    #     )
    #     torch.save(checkpoint, path)
    #     self.logger.info(f"Checkpoint saved: {path}")
    
    # def _load_checkpoint(self, name):
    #     """加载检查点"""
    #     path = os.path.join(
    #         self.config.checkpoint_dir,
    #         f"{self.config.experiment_name}_{name}.pth"
    #     )
        
    #     if os.path.exists(path):
    #         checkpoint = torch.load(path, map_location=self.device)
    #         self.model.load_state_dict(checkpoint['model_state_dict'])
    #         self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #         self.best_threshold = checkpoint.get('best_threshold', 0.5)
    #         self.best_val_f1 = checkpoint.get('best_val_f1', 0)
    #         self.best_val_metrics = checkpoint.get('best_val_metrics', {})
    #         # self.best_test_metrics = checkpoint.get('best_test_metrics', {})
    #         self.logger.info(f"Checkpoint loaded: {path}")
    #     else:
    #         self.logger.warning(f"Checkpoint not found: {path}")

    def check_connectivity_with_nx(slef, graphs, all_outputs, threshold):
        """
        使用NetworkX判断导出子图连通性
        """
        import networkx as nx
        
        
        for g in range(len(all_outputs)):
            # 转换为numpy
            edge_index = graphs[g]['graph']['edge_index']
            label = graphs[g]['target']['community_mask']
            comm_size = len(graphs[g]['target']['community_indices'])
            print(f'图节点数：{edge_index.max().item() + 1}')
            predictions = all_outputs[g]
            
            if torch.is_tensor(edge_index):
                edge_index = edge_index.cpu().numpy()

            # 选择节点
            selected_nodes = set(np.where(predictions == 1)[0])
            
            print(f"选择的节点: {selected_nodes}")
            print(f"选择的节点数: {len(selected_nodes)}")
            print(f"社区: {sorted(graphs[g]['target']['community_indices'])}")
            
            # # 构建完整图
            # G_full = nx.Graph()
            # for i in range(edge_index.shape[1]):
            #     u, v = edge_index[0, i], edge_index[1, i]
            #     G_full.add_edge(u, v)
            
            # # 提取导出子图
            # subgraph = G_full.subgraph(selected_nodes)
            
            # # 检查连通性
            # is_connected = nx.is_connected(subgraph)
            # num_components = nx.number_connected_components(subgraph)
            
            # # print(f"子图节点数: {subgraph.number_of_nodes()}")
            
            # # print(f"连通分量数: {num_components}")
            
            # if not is_connected:
            #     components = list(nx.connected_components(subgraph))
            #     for i, comp in enumerate(components):
            #         print(f"  分量 {i+1}: {sorted(comp)}")
def load_query_batches(save_dir, dataset_name):
    """
    从文件加载 support_batchs 和 query_batchs
    
    Args:
        save_dir: 保存目录
        dataset_name: 数据集名称
        num_shots: few-shot 数量
        
    Returns:
        support_batchs, query_batchs
    """
    load_path = os.path.join(save_dir, f"{dataset_name}_queries.pt")
    
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"文件不存在: {load_path}")
    
    loaded_data = torch.load(load_path)

    query_batchs = loaded_data['query_batchs']
    
    print(f"从 {load_path} 加载数据成功")
    print(f"Graphs数量: {len(query_batchs)}")
    print(f"第一个Graph的query大小: {len(query_batchs[0]) if query_batchs else 0}")
    
    return query_batchs
def update_config_from_args(args):
    """从命令行参数更新配置"""
    # 数据集名
    if args.dataset:
        config.dataset_name = args.dataset
        print(f"  Using dataset: {config.dataset_name}")
    
    # 流层数（flow layers）
    if args.n_flows is not None:
        config.n_flows = args.n_flows
        print(f"  Setting flow layers: {config.n_flows}")
    if args.latent_steps is not None:
        config.latent_steps = args.latent_steps
        print(f"  Setting latent steps: {config.latent_steps}")
    if args.tau is not None:
        config.kl_threshold = args.tau
        print(f"  Setting kl threshold: {config.kl_threshold}")
    if args.zeta is not None:
        config.prob_threshold = args.zeta
        print(f"  Setting Probability threshold for optimization: {config.prob_threshold}")
    if args.lambda_reg is not None:
        config.latent_reg = args.lambda_reg
        print(f"  Setting Weight of region loss: {config.latent_reg}")
    if args.lambda_vol is not None:
        config.latent_volume = args.lambda_vol
        print(f"  Setting Weight of volume loss: {config.latent_volume}")
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
        print(f"  Setting learning rate: {config.learning_rate}")
    
    


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='LSO-CS: Community Search with GCN-style Training')
    
    # 数据集参数
    parser.add_argument('--dataset', type=str, default=None,
                        help='Dataset name (e.g., facebook, twitter, citeseer, cora)')
    
    # 模型结构参数
    parser.add_argument('--n_flows', type=int, default=None,
                        help='Number of flow layers (default: from config)')
    parser.add_argument('--latent_steps', type=int, default=None,
                        help='Number of optimization step (default: from config)')
    parser.add_argument('--attr', type=int, default=1,
                        help='Whether to use query attributes (default: 1)')
    parser.add_argument('--tau', type=float, default=0.1,
                        help='Threshold for KL divergence (default: 0.1)')
    parser.add_argument('--zeta', type=float, default=0.1,
                        help='Probability threshold for optimization (default: 0.1)')
    parser.add_argument('--lambda_reg', type=float, default=1,
                        help='Weight of region loss (default: 1)')
    parser.add_argument('--lambda_vol', type=float, default=0.5,
                        help='Weight of volume loss (default: 0.5)')
    parser.add_argument('--learning_rate', type=float, default=0.0005,
                        help='learning rate (default: 0.0005)')
    
    
    
    return parser.parse_args()
def main():
    # 解析命令行参数
    args = parse_arguments()
    
    # 更新配置
    update_config_from_args(args)
    # use_query_attr = True
    if args.attr is not None:
        use_query_attr = bool(args.attr)
    print(f"  Using query attributes: {use_query_attr}")
    """主函数：基于GCN风格的训练流程"""
    # print("=" * 60)
    # print("LSO-CS: Community Search with GCN-style Training")
    # print("=" * 60)
    for key, value in vars(config).items():
        print(f"{key}: {value}")
    # 1. 加载数据集
    print("\n[1/5] Loading dataset...")
    if config.dataset_name in ['facebook', 'twitter']:
        queries = load_query_batches(config.data_dir, config.dataset_name)
        gn = len(queries)
        train_queries, valid_queries, test_queries = queries[0:int(gn*0.6)], queries[int(gn*0.6):int(gn*0.8)], queries[int(gn*0.8):]
        # train_queries, valid_queries, test_queries = queries[0:int(gn*0.2)], queries[int(gn*0.2):int(gn*0.4)], queries[int(gn*0.4):int(gn*0.6)]
    
        train_loader = []
        val_loader = []
        test_loader = []
        # train_loader.extend(train_queries[0])
        # val_loader.extend(valid_queries[0])
        # test_loader.extend(test_queries[0])
        for q in train_queries:
            train_loader.extend(q)
        for q in valid_queries:
            val_loader.extend(q)
        for q in test_queries:
            test_loader.extend(q)
    elif config.dataset_name in ['twitter2facebook']:
        queries = load_query_batches(config.data_dir, config.dataset_name)
        gn = len(queries)
        train_queries, valid_queries, test_queries = queries[0:140], queries[140:195], queries[195:]
    
        train_loader = []
        val_loader = []
        test_loader = []
        for q in train_queries:
            train_loader.extend(q)
        for q in valid_queries:
            val_loader.extend(q)
        for q in test_queries:
            test_loader.extend(q)
    else:
        queries = load_query_batches(config.data_dir, config.dataset_name)[0]
        gn = len(queries)
        train_loader, val_loader, test_loader = queries[0:int(gn*0.6)], queries[int(gn*0.6):int(gn*0.8)], queries[int(gn*0.8):]

    print(f"  Dataset: {config.dataset_name}")
    print(f"  Train: {len(train_loader)} samples")
    print(f"  Validation: {len(val_loader)} samples")
    print(f"  Test: {len(test_loader)} samples")
    
    # 2. 创建模型
    print("\n[2/5] Creating model...")
    model = LSOCommunitySearch(config)
    print(model)
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 3. 创建训练器
    print("\n[3/5] Creating trainer...")
    trainer = Trainer(model, config)
    
    # 4. 训练循环
    print("\n[4/5] Starting training...")
    print("-" * 60)
    
    train_counter = 0
    train_time_total = 0
    test_time = 0
    for epoch in range(1, config.num_epochs + 1):
        epoch_start_time = time.time()
        
        # 调整学习率（多项式衰减）
        current_lr = trainer.adjust_lr_poly(epoch, config.num_epochs)
        
        # 训练
        train_start = time.time()
        train_loss, train_metrics = trainer.train_epoch(train_loader, epoch, use_query_attr)
        epoch_train_time = time.time() - train_start
        print(f"train time for epoch {epoch}: {epoch_train_time:.4f}s")
        train_time_total = train_time_total + (epoch_train_time)
        train_counter += 1
        # 每10个epoch或在最后一个epoch进行验证
        # if epoch % 10 == 0 or epoch == config.num_epochs:
            # 在验证集上评估并搜索最佳阈值
        best_threshold, val_f1, val_metrics = trainer.evaluate_on_val(val_loader, epoch, use_query_attr)
        # print(val_metrics)
        # print(trainer.best_val_metrics)
        # 
        # 
        # 保存最佳模型
        # test_metrics = {'avg_f1': 0.0}
        if val_f1 > trainer.best_val_f1:
            trainer.early_stop_counter = 0
            trainer.best_val_f1 = val_f1
            trainer.best_threshold = best_threshold
            trainer.best_val_metrics = val_metrics
            test_start = time.time()
            test_metrics, all_predictions = trainer.test_with_best_threshold(test_loader, best_threshold, use_query_attr)
            test_time = time.time() - test_start
            trainer.best_test_f1 = test_metrics["avg_f1"]
            # 用最佳阈值测试
            
            # trainer.check_connectivity_with_nx(test_loader, all_predictions, trainer.best_threshold)
            trainer.best_test_metrics = test_metrics
            trainer.all_predictions = all_predictions
            
            # 保存检查点
            # trainer._save_checkpoint(epoch, val_metrics, 'best')
            
            # trainer.logger.info(f"New best model at epoch {epoch}: "
            #                   f"Val F1={val_f1:.4f}, "
            #                   f"Test F1={test_metrics.get('avg_f1', 0):.4f}"
            #                   )
        else:
            trainer.early_stop_counter += 1
        # 5. 早停检查
        if trainer.early_stop_counter >= config.patience:
            print(f"\n[Early Stop] No improvement for {config.patience} epochs. Stopping.")
            break
        
        
        # 打印进度
        epoch_time = time.time() - epoch_start_time
        lr_info = f"LR: {current_lr:.2e}"
        
        # if epoch % 10 == 0:
        print(f"Epoch {epoch:03d}/{config.num_epochs:03d} | "
                f"Time: {epoch_time:.1f}s | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val F1: {val_f1:.4f} | "
                f"Val Pre: {val_f1:.4f} | "
                f"Best Thr: {best_threshold:.3f} | "
                f"Test F1: {test_metrics['avg_f1']:.4f} |"
                f"{lr_info}")
        # else:
        #     print(f"Epoch {epoch:03d}/{config.num_epochs:03d} | "
        #           f"Time: {epoch_time:.1f}s | "
        #           f"Train Loss: {train_loss:.4f} | "
        #           f"{lr_info}")
    

    # 5. 最终测试
    print("\n[5/5] Final testing with best threshold...")
    print("-" * 60)
    
    # 加载最佳模型
    # trainer._load_checkpoint('best')
    
    # 最终测试
    # final_metrics, all_predictions = trainer.test_with_best_threshold(test_loader, trainer.best_threshold)
    trainer.check_connectivity_with_nx(test_loader, trainer.all_predictions, trainer.best_threshold)
    # 打印最终结果
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Best Threshold: {trainer.best_threshold:.3f}")
    print(f"Best Validation F1: {trainer.best_val_f1:.4f}")
    print("\nTest Metrics:")
    # print(f"  F1 Score: {trainer.best_test_metrics.get('avg_f1', 0):.4f} ± {trainer.best_test_metrics.get('std_f1', 0):.4f}")
    # print(f"  Precision: {trainer.best_test_metrics.get('avg_precision', 0):.4f} ± {trainer.best_test_metrics.get('std_precision', 0):.4f}")
    # print(f"  Recall: {trainer.best_test_metrics.get('avg_recall', 0):.4f} ± {trainer.best_test_metrics.get('std_recall', 0):.4f}")
        
    metric_display = {
        'f1': 'F1 Score',
        'precision': 'Precision',
        'recall': 'Recall',
        'jaccard': 'Jaccard',
        'nmi': 'NMI',
        'ari': 'ARI'
    }
    
    for metric_key, display_name in metric_display.items():
        avg_key = f'avg_{metric_key}'
        std_key = f'std_{metric_key}'
        if avg_key in trainer.best_test_metrics:
            print(f"{display_name:>12}: {trainer.best_test_metrics[avg_key]:.4f} ± {trainer.best_test_metrics[std_key]:.4f}")
        # for metric_key, display_name in metric_display.items():
        #     avg_key = f'avg_{metric_key}'
        #     std_key = f'std_{metric_key}'
        #     if avg_key in final_metrics:
        #         print(f"{display_name:>12}: {final_metrics[avg_key]:.4f} ± {final_metrics[std_key]:.4f}")
    print(f"Average train time: {train_time_total/train_counter:.4f}")
    print(f"test time: {test_time:.4f}")    
    print("\n" + "=" * 60)
    print("All tasks completed!")
    # print(f"Results saved to: {config.result_dir}")
    # print(f"Logs saved to: {config.log_dir}")
    # print(f"Checkpoints saved to: {config.checkpoint_dir}")
    print("=" * 60)
    return trainer.best_test_metrics.get('avg_f1', 0), trainer.best_test_metrics.get('avg_precision', 0), trainer.best_test_metrics.get('avg_recall', 0)

def set_seed():
        """设置随机种子确保可复现性"""
        import random
        import numpy as np
        
        random.seed(config.random_seed)
        np.random.seed(config.random_seed)
        torch.manual_seed(config.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.random_seed)
            torch.cuda.manual_seed_all(config.random_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
if __name__ == "__main__":
    F1lists = []
    Prelists = []
    Reclists = []
    for i in range(5):
    # for i in range(1):
        set_seed()
        F1, Pre, Rec = main()
        F1lists.append(F1)
        Prelists.append(Pre)
        Reclists.append(Rec)
    print(f"F1: {np.mean(F1lists):.4f}")
    print(f"F1 std: {np.std(F1lists):.4f}")
    print(f"Precision: {np.mean(Prelists):.4f}")
    print(f"Precision std: {np.std(Prelists):.4f}")
    print(f"Recall: {np.mean(Reclists):.4f}")
    print(f"Recall std: {np.std(Reclists):.4f}")