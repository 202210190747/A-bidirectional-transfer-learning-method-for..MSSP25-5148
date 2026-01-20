# A-bidirectional-transfer-learning-method-for..MSSP25-5148
Includes the code, public datasets, and robot milling data

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime

# -------------------------- 1. 核心参数配置 --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

datasets = {
    'C1': 'filtered_data_normalized-C1.csv',
    'C4': 'filtered_data_normalized-C4.csv',
    'C6': 'filtered_data_normalized-C6.csv',
    'RC1': 'filtered_data_normalized-ROBOT-C1.csv',
    'RC2': 'filtered_data_normalized-ROBOT-C2.csv',
    'RC3': 'filtered_data_normalized-ROBOT-C3.csv'
}


experiments = [
    {'name': 'C1+C4+C6→RC1', 'source_keys': ['C1', 'C4', 'C6'], 'target_key': 'RC1'},
    {'name': 'C1+C4+C6→RC2', 'source_keys': ['C1', 'C4', 'C6'], 'target_key': 'RC2'},
    {'name': 'C1+C4+C6→RC3', 'source_keys': ['C1', 'C4', 'C6'], 'target_key': 'RC3'},
]

# 学习方法列表（保持不变，与原代码一致）
methods = [
    'fine_tuning',          # 迁移：微调
    'feature_extraction',   # 迁移：特征提取
    'adversarial',          # 迁移：对抗性学习
    'tcn_bigru_attention',  # 仅源域：TCN-BiGRU-Attention
    'cnn_lstm',             # 仅源域：CNN-LSTM
    'transformer'           # 仅源域：Transformer
]

# 训练参数（保持不变，与原代码一致） 参数可以使用opt进行优化
source_epochs = 25      # 这个地方的设置可以按需设置
target_epochs = 25       
batch_size = 64           
learning_rate = 0.001     # 快速跑通
alpha = 0.01              # PINN物理损失权重
beta = 0.01               # 单调性损失权重
lambda_adv = 0.1          # 对抗损失权重
loss_fn = nn.MSELoss()    # 回归损失函数


# -------------------------- 2. 数据处理工具函数 --------------------------
# （完全不变，与原代码一致）
def load_single_dataset(dataset_key):
    try:
        data = pd.read_csv(datasets[dataset_key])
        if data.shape[1] < 2:
            raise ValueError(f"数据集{dataset_key}格式错误，需至少1特征+1标签")
        print(f"✅ 加载数据集: {datasets[dataset_key]} | 样本数: {len(data)} | 特征数: {data.shape[1]-1}")
        return data
    except FileNotFoundError:
        print(f"❌ 未找到数据集文件: {datasets[dataset_key]}")
        exit()

def add_time_feature(data):
    if 'Time' not in data.columns:
        data.insert(0, 'Time', np.linspace(0, 1, len(data)))  # 时间特征归一化到0~1
    return data

def merge_source_datasets(source_keys):
    merged_X, merged_y = None, None
    for key in source_keys:
        data = load_single_dataset(key)
        data = add_time_feature(data)
        X = data.iloc[:, :-1].values  # 特征（含时间）：[样本数, 特征数]
        y = data.iloc[:, -1].values   # 标签（RUL）：[样本数]
        if merged_X is None:
            merged_X, merged_y = X, y
        else:
            merged_X = np.concatenate([merged_X, X], axis=0)
            merged_y = np.concatenate([merged_y, y], axis=0)
    print(f"📊 合并源域 {source_keys} | 总样本数: {len(merged_X)} | 特征数: {merged_X.shape[1]}")
    return merged_X, merged_y

def split_target_data(X_target, y_target):
    """目标域划分：测试集（每隔5选1）+ 微调集（剩余每隔8选1）"""
    total_indices = np.arange(len(X_target))
    
    # 测试集（~20%）
    test_indices = total_indices[::5]
    X_test = X_target[test_indices]
    y_test = y_target[test_indices]
    
    # 微调集（从剩余样本中选）
    remaining_indices = np.setdiff1d(total_indices, test_indices)
    if len(remaining_indices) < 8:  # 避免样本数不足导致无数据
        ft_indices = remaining_indices
    else:
        ft_indices = remaining_indices[::8]
    
    X_ft = X_target[ft_indices]
    y_ft = y_target[ft_indices]
    
    print(f"🎯 目标域采样 | 微调集: {len(X_ft)} 样本 | 测试集: {len(X_test)} 样本")
    return X_ft, y_ft, X_test, y_test

def prepare_tensors(data, device):
    """转换为时序模型标准格式：[样本数, seq_len=1, 特征数]"""
    tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(1)  # 新增seq_len维度
    return tensor.to(device)  # 最终形状：[batch, seq_len=1, features]


# -------------------------- 3. 模型定义 --------------------------
# （仅BaseModel显式添加hidden_size参数，其余保持不变）
class ChannelAttention(nn.Module):
    """通道注意力机制"""
    def __init__(self, hidden_size, reduction_ratio=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size//reduction_ratio, hidden_size, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [batch, seq_len, hidden_size]
        b, l, c = x.size()
        y = self.avg_pool(x.permute(0,2,1)).view(b, c)  # 通道全局池化
        y = self.fc(y).view(b, 1, c)                    # 生成通道权重
        return x * y.expand_as(x)                       # 施加注意力权重

class BaseModel(nn.Module):
    """基础模型（用于迁移学习）- 显式添加hidden_size参数"""
    def __init__(self, input_size, hidden_size=64):  # 显式定义hidden_size，默认64
        super().__init__()
        self.hidden_size = hidden_size  # 记录隐藏层大小，便于后续参数保存
        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=True,
            batch_first=True
        )
        self.channel_attn = ChannelAttention(hidden_size*2)  # 双向输出维度翻倍
        self.fc = nn.Sequential(
            nn.Linear(hidden_size*2, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()  # RUL归一化到0~1
        )
        self.time_idx = 0  # 时间特征索引

    def forward(self, x):
        # x: [batch, seq_len, features]
        x, _ = self.bilstm(x)  # [batch, seq_len, hidden_size*2]
        x = self.channel_attn(x)
        return self.fc(x[:, -1, :])  # 取最后时间步输出

    def physics_loss(self, y_pred, x):
        """PINN物理约束：RUL随时间递减"""
        t = x[:, :, self.time_idx].squeeze()  # 提取时间特征
        return torch.mean(torch.relu(y_pred.squeeze() - (1 - t)))

    def monotonicity_loss(self, y_pred, x):
        """单调性约束：RUL随时间单调递减"""
        t = x[:, :, self.time_idx].squeeze()
        sorted_idx = torch.argsort(t)
        sorted_pred = y_pred[sorted_idx]
        diffs = sorted_pred[1:] - sorted_pred[:-1]  # 后值-前值≤0
        return torch.mean(torch.relu(diffs))

    def feature_extractor(self, x):
        """特征提取器（用于迁移学习）"""
        x, _ = self.bilstm(x)
        return self.channel_attn(x)

class TCNLayer(nn.Module):
    """TCN层（适配短序列）"""
    def __init__(self, in_channels, out_channels, kernel_size=2, dilation=1):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=0,
            dilation=dilation
        )
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

    def forward(self, x):
        # x: [batch, in_channels, seq_len]
        residual = x
        
        # 若seq_len < 核大小，用padding补全
        if x.size(2) < self.conv.kernel_size[0]:
            pad_size = self.conv.kernel_size[0] - x.size(2)
            x = nn.functional.pad(x, (0, pad_size))  # 右侧补0
        
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        
        # 残差连接维度对齐
        if self.residual is not None:
            residual = self.residual(residual)
        if x.size(2) != residual.size(2):
            residual = nn.functional.adaptive_avg_pool1d(residual, x.size(2))
        
        return x + residual

class TCNBiGRUAttention(nn.Module):
    """TCN-BiGRU-Attention模型"""
    def __init__(self, input_size, tcn_channels=[32, 64], gru_hidden=64, num_heads=2):
        super().__init__()
        self.tcn_input_proj = nn.Linear(input_size, tcn_channels[0])  # 特征投影
        tcn_layers = []
        in_channels = tcn_channels[0]
        for out_channels in tcn_channels[1:]:
            tcn_layers.append(TCNLayer(in_channels, out_channels))
            in_channels = out_channels
        self.tcn = nn.Sequential(*tcn_layers)
        
        self.bigru = nn.GRU(
            input_size=tcn_channels[-1],
            hidden_size=gru_hidden,
            bidirectional=True,
            batch_first=True
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=gru_hidden*2,
            num_heads=num_heads,
            batch_first=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(gru_hidden*2, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [batch, seq_len, features]
        x_proj = self.tcn_input_proj(x)  # [batch, seq_len, tcn_channels[0]]
        x_tcn = x_proj.permute(0, 2, 1)  # [batch, channels, seq_len]
        x_tcn = self.tcn(x_tcn)          # [batch, tcn_channels[-1], seq_len]
        
        x_gru = x_tcn.permute(0, 2, 1)   # [batch, seq_len, tcn_channels[-1]]
        x_gru, _ = self.bigru(x_gru)     # [batch, seq_len, gru_hidden*2]
        
        attn_output, _ = self.attention(x_gru, x_gru, x_gru)  # 自注意力
        x_out = attn_output[:, -1, :]    # 取最后时间步
        
        return self.fc(x_out)  # [batch, 1]

class CNNLSTM(nn.Module):
    """CNN-LSTM模型（修复池化问题）"""
    def __init__(self, input_size, cnn_filters=[32, 64], lstm_hidden=64):
        super().__init__()
        # CNN部分（移除池化层）
        cnn_layers = []
        in_channels = input_size
        for out_channels in cnn_filters:
            cnn_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=1),  # 1x1卷积
                nn.BatchNorm1d(out_channels),
                nn.ReLU()
            ])
            in_channels = out_channels
        self.cnn = nn.Sequential(*cnn_layers)
        
        self.lstm = nn.LSTM(
            input_size=cnn_filters[-1],
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            dropout=0.2
        )
        
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [batch, seq_len, features]
        x_cnn = x.permute(0, 2, 1)  # [batch, features, seq_len]
        x_cnn = self.cnn(x_cnn)     # [batch, cnn_filters[-1], seq_len]
        
        x_lstm = x_cnn.permute(0, 2, 1)  # [batch, seq_len, cnn_filters[-1]]
        x_lstm, _ = self.lstm(x_lstm)    # [batch, seq_len, lstm_hidden]
        
        x_out = x_lstm[:, -1, :]         # 取最后时间步
        return self.fc(x_out)  # [batch, 1]

class TransformerModel(nn.Module):
    """Transformer模型"""
    def __init__(self, input_size, d_model=64, nhead=2, num_layers=1):
        super().__init__()
        self.proj = nn.Linear(input_size, d_model)  # 特征投影
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=0.2,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [batch, seq_len, features]
        x_proj = self.proj(x)  # [batch, seq_len, d_model]
        x_enc = self.transformer_encoder(x_proj)  # [batch, seq_len, d_model]
        x_out = x_enc[:, -1, :]  # 取最后时间步
        return self.fc(x_out)  # [batch, 1]


# -------------------------- 4. 迁移学习组件与训练方法 --------------------------
# （完全不变，与原代码一致）
class GradientReversalLayer(nn.Module):
    """梯度反转层（对抗性迁移）"""
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return x

    def backward(self, grad_output):
        return grad_output * (-self.lambda_)

class DomainClassifier(nn.Module):
    """域分类器（对抗性迁移）"""
    def __init__(self, input_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_avg = torch.mean(x, dim=1)  # [batch, features]
        return self.fc(x_avg)

def evaluate_model(model, X_tensor, y_tensor):
    """模型评估函数"""
    model.eval()
    with torch.no_grad():
        y_pred = model(X_tensor).cpu().numpy().flatten()
        y_true = y_tensor.cpu().numpy().flatten()
    
    return {
        'true_rul': y_true,
        'pred_rul': y_pred,
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred)
    }

def train_source_only_model(model, X_train, y_train, X_test, y_test, device, model_name):
    """仅源域训练模型"""
    print(f"\n📌 {model_name} 源域训练（无目标域微调）")
    dataset = TensorDataset(X_train, y_train.unsqueeze(1))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    start_time = time.time()
    for epoch in range(source_epochs):
        model.train()
        total_loss = 0.0
        
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            y_pred = model(batch_x)
            loss = loss_fn(y_pred, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)
        
        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / len(loader.dataset)
            eval_res = evaluate_model(model, X_test, y_test.unsqueeze(1))
            print(f"  Epoch [{epoch+1}/{source_epochs}] | Loss: {avg_loss:.4f} | Test RMSE: {eval_res['rmse']:.4f}")
    
    train_time = time.time() - start_time
    final_res = evaluate_model(model, X_test, y_test.unsqueeze(1))
    print(f"📌 {model_name} 训练完成 | 耗时: {train_time:.2f} 秒")
    return final_res, train_time

def pretrain_source_model(model, X_train, y_train, device):
    """源域预训练（迁移学习共用）"""
    dataset = TensorDataset(X_train, y_train.unsqueeze(1))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    start_time = time.time()
    for epoch in range(source_epochs):
        model.train()
        total_loss = 0.0
        
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            y_pred = model(batch_x)
            
            data_loss = loss_fn(y_pred, batch_y)
            pde_loss = model.physics_loss(y_pred, batch_x)
            mono_loss = model.monotonicity_loss(y_pred, batch_x)
            loss = data_loss + alpha*pde_loss + beta*mono_loss
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)
        
        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / len(loader.dataset)
            print(f"  源域预训练 Epoch [{epoch+1}/{source_epochs}] | Loss: {avg_loss:.4f}")
    
    pretrain_time = time.time() - start_time
    print(f"📌 源域预训练完成 | 耗时: {pretrain_time:.2f} 秒")
    return model, pretrain_time

# 迁移学习训练方法（保持不变）
def train_fine_tuning(source_model, target_model, X_ft, y_ft, X_test, y_test, device):
    print("\n📌 微调迁移学习训练")
    target_model.load_state_dict(source_model.state_dict())
    dataset = TensorDataset(X_ft, y_ft.unsqueeze(1))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.AdamW(target_model.parameters(), lr=learning_rate/10, weight_decay=1e-4)
    
    start_time = time.time()
    for epoch in range(target_epochs):
        target_model.train()
        total_loss = 0.0
        
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            y_pred = target_model(batch_x)
            data_loss = loss_fn(y_pred, batch_y)
            pde_loss = target_model.physics_loss(y_pred, batch_x)
            mono_loss = target_model.monotonicity_loss(y_pred, batch_x)
            loss = data_loss + alpha*pde_loss + beta*mono_loss
            
            loss.backward()
            nn.utils.clip_grad_norm_(target_model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)
        
        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / len(loader.dataset)
            eval_res = evaluate_model(target_model, X_test, y_test.unsqueeze(1))
            print(f"  Epoch [{epoch+1}/{target_epochs}] | Loss: {avg_loss:.4f} | Test RMSE: {eval_res['rmse']:.4f}")
    
    ft_time = time.time() - start_time
    final_res = evaluate_model(target_model, X_test, y_test.unsqueeze(1))
    return final_res, ft_time

def train_feature_extraction(source_model, target_model, X_ft, y_ft, X_test, y_test, device):
    print("\n📌 特征提取迁移学习训练")
    target_model.load_state_dict(source_model.state_dict())
    # 冻结特征层
    for param in target_model.bilstm.parameters():
        param.requires_grad = False
    for param in target_model.channel_attn.parameters():
        param.requires_grad = False
    
    dataset = TensorDataset(X_ft, y_ft.unsqueeze(1))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.AdamW(target_model.fc.parameters(), lr=learning_rate/5, weight_decay=1e-4)
    
    start_time = time.time()
    for epoch in range(target_epochs):
        target_model.train()
        total_loss = 0.0
        
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            y_pred = target_model(batch_x)
            data_loss = loss_fn(y_pred, batch_y)
            pde_loss = target_model.physics_loss(y_pred, batch_x)
            mono_loss = target_model.monotonicity_loss(y_pred, batch_x)
            loss = data_loss + alpha*pde_loss + beta*mono_loss
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)
        
        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / len(loader.dataset)
            eval_res = evaluate_model(target_model, X_test, y_test.unsqueeze(1))
            print(f"  Epoch [{epoch+1}/{target_epochs}] | Loss: {avg_loss:.4f} | Test RMSE: {eval_res['rmse']:.4f}")
    
    ft_time = time.time() - start_time
    final_res = evaluate_model(target_model, X_test, y_test.unsqueeze(1))
    return final_res, ft_time

def train_mmd_transfer(source_model, target_model, X_source, y_source, X_ft, y_ft, X_test, y_test, device):
    print("\n📌 MMD迁移学习训练")
    target_model.load_state_dict(source_model.state_dict())
    source_dataset = TensorDataset(X_source, y_source.unsqueeze(1))
    source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True)
    target_dataset = TensorDataset(X_ft, y_ft.unsqueeze(1))
    target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = optim.AdamW(target_model.parameters(), lr=learning_rate/10, weight_decay=1e-4)
    start_time = time.time()
    target_iter = iter(target_loader)
    
    for epoch in range(target_epochs):
        target_model.train()
        total_loss = 0.0
        
        for batch_x, batch_y in source_loader:
            try:
                target_x, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                target_x, _ = next(target_iter)
            
            min_size = min(batch_x.size(0), target_x.size(0))
            batch_x, batch_y = batch_x[:min_size], batch_y[:min_size]
            target_x = target_x[:min_size]
            
            optimizer.zero_grad()
            y_pred = target_model(batch_x)
            
            data_loss = loss_fn(y_pred, batch_y)
            pde_loss = target_model.physics_loss(y_pred, batch_x)
            mono_loss = target_model.monotonicity_loss(y_pred, batch_x)
            
            source_feat = target_model.feature_extractor(batch_x)
            target_feat = target_model.feature_extractor(target_x)
            mmd = mmd_loss(source_feat, target_feat)
            
            loss = data_loss + alpha*pde_loss + beta*mono_loss + gamma*mmd
            loss.backward()
            nn.utils.clip_grad_norm_(target_model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)
        
        if (epoch + 1) % 5 == 0:
            eval_res = evaluate_model(target_model, X_test, y_test.unsqueeze(1))
            print(f"  Epoch [{epoch+1}/{target_epochs}] | MMD: {mmd.item():.4f} | Test RMSE: {eval_res['rmse']:.4f}")
    
    ft_time = time.time() - start_time
    final_res = evaluate_model(target_model, X_test, y_test.unsqueeze(1))
    return final_res, ft_time

def train_adversarial(source_model, target_model, X_source, y_source, X_ft, y_ft, X_test, y_test, device):
    print("\n📌 对抗性迁移学习训练")
    target_model.load_state_dict(source_model.state_dict())
    domain_clf = DomainClassifier(input_dim=128).to(device)
    grl = GradientReversalLayer(lambda_=lambda_adv)
    
    source_dataset = TensorDataset(X_source, y_source.unsqueeze(1))
    source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True)
    target_dataset = TensorDataset(X_ft, y_ft.unsqueeze(1))
    target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True)
    
    feat_optimizer = optim.AdamW(target_model.parameters(), lr=learning_rate/10, weight_decay=1e-4)
    clf_optimizer = optim.AdamW(domain_clf.parameters(), lr=learning_rate/10, weight_decay=1e-4)
    domain_criterion = nn.BCELoss()
    
    start_time = time.time()
    target_iter = iter(target_loader)
    
    for epoch in range(target_epochs):
        target_model.train()
        domain_clf.train()
        
        for batch_x, batch_y in source_loader:
            try:
                target_x, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                target_x, _ = next(target_iter)
            
            min_size = min(batch_x.size(0), target_x.size(0))
            batch_x, batch_y = batch_x[:min_size], batch_y[:min_size]
            target_x = target_x[:min_size]
            
            source_labels = torch.zeros(min_size, 1).to(device)
            target_labels = torch.ones(min_size, 1).to(device)
            
            # 训练域分类器
            clf_optimizer.zero_grad()
            source_feat = target_model.feature_extractor(batch_x)
            target_feat = target_model.feature_extractor(target_x)
            
            source_pred = domain_clf(grl(source_feat.detach()))
            target_pred = domain_clf(grl(target_feat.detach()))
            clf_loss = domain_criterion(source_pred, source_labels) + domain_criterion(target_pred, target_labels)
            clf_loss.backward()
            clf_optimizer.step()
            
            # 训练特征提取器
            feat_optimizer.zero_grad()
            y_pred = target_model(batch_x)
            
            data_loss = loss_fn(y_pred, batch_y)
            pde_loss = target_model.physics_loss(y_pred, batch_x)
            mono_loss = target_model.monotonicity_loss(y_pred, batch_x)
            
            source_pred = domain_clf(grl(source_feat))
            target_pred = domain_clf(grl(target_feat))
            adv_loss = domain_criterion(source_pred, target_labels) + domain_criterion(target_pred, source_labels)
            
            loss = data_loss + alpha*pde_loss + beta*mono_loss + lambda_adv*adv_loss
            loss.backward()
            nn.utils.clip_grad_norm_(target_model.parameters(), 1.0)
            feat_optimizer.step()
        
        if (epoch + 1) % 5 == 0:
            eval_res = evaluate_model(target_model, X_test, y_test.unsqueeze(1))
            print(f"  Epoch [{epoch+1}/{target_epochs}] | Adv Loss: {adv_loss.item():.4f} | Test RMSE: {eval_res['rmse']:.4f}")
    
    ft_time = time.time() - start_time
    final_res = evaluate_model(target_model, X_test, y_test.unsqueeze(1))
    return final_res, ft_time


# -------------------------- 5. 结果保存与可视化 --------------------------
# （新增：1. 文件夹名称含参数；2. 网络参数CSV保存函数）
def create_results_root():
    """修改：生成包含关键网络参数的结果文件夹名称"""
    # 1. 定义所有模型的关键结构参数（与模型__init__默认值对齐）
    base_hidden = 64          # BaseModel的LSTM隐藏层大小
    tcn_channels = [32, 64]   # TCN-BiGRU-Attention的TCN通道数
    gru_hidden = 64           # TCN-BiGRU-Attention的GRU隐藏层大小
    cnn_filters = [32, 64]    # CNN-LSTM的CNN滤波器数
    trans_d_model = 64        # Transformer的d_model
    
    # 2. 格式化参数字符串（避免特殊字符，确保文件夹名合法）
    param_str = (
        f"hid{base_hidden}_"
        f"tcn{'-'.join(map(str, tcn_channels))}_"
        f"gru{gru_hidden}_"
        f"cnn{'-'.join(map(str, cnn_filters))}_"
        f"trans{trans_d_model}"
    )
    
    # 3. 生成最终文件夹名：前缀_参数_时间戳
    time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    root_dir = f"transfer_comparison_{param_str}_{time_str}"
    os.makedirs(root_dir, exist_ok=True)
    
    return root_dir, param_str  # 返回参数字符串，便于后续日志打印

def save_prediction_csv(root_dir, target_key, method, eval_res):
    """（保持不变）保存预测结果CSV"""
    sorted_idx = np.argsort(eval_res['true_rul'])[::-1]
    sorted_true = eval_res['true_rul'][sorted_idx]
    sorted_pred = eval_res['pred_rul'][sorted_idx]
    abs_error = np.abs(sorted_true - sorted_pred)
    
    model_name = method.replace('_', ' ').title()
    df = pd.DataFrame({
        'Dataset': [target_key] * len(sorted_true),
        'Model': [model_name] * len(sorted_true),
        'True_RUL': sorted_true,
        'Predicted_RUL': sorted_pred,
        'Absolute_Error': abs_error
    })
    
    save_path = os.path.join(root_dir, f"{target_key}_{method}_predictions.csv")
    df.to_csv(save_path, index=False)
    return save_path

def save_performance_csv(root_dir, performance_list, is_init=False):
    """（保持不变）保存性能汇总表"""
    df = pd.DataFrame(performance_list)
    save_path = os.path.join(root_dir, "all_models_performance.csv")
    
    if is_init:
        df.to_csv(save_path, index=False, mode='w')
    else:
        df.to_csv(save_path, index=False, mode='a', header=False)
    return save_path

def save_network_params_csv(root_dir, network_params_list):
    """新增：保存网络模型结构与训练参数到CSV"""
    # 定义CSV列名（覆盖所有模型的参数维度）
    columns = [
        "Experiment_Name",       # 实验名称（如RC1+RC2+RC3→C1）
        "Model_Method",          # 模型方法（如fine_tuning、tcn_bigru_attention）
        "Model_Type",            # 模型类型（如BaseModel、TCNBiGRUAttention）
        "Input_Size",            # 输入特征数（动态获取）
        # BaseModel参数
        "BaseModel_HiddenSize",  # BaseModel的LSTM隐藏层大小
        # TCN-BiGRU-Attention参数
        "TCN_Channels",          # TCN通道数（逗号分隔，如"32,64"）
        "GRU_Hidden",            # GRU隐藏层大小
        "GRU_NumHeads",          # 注意力头数
        # CNN-LSTM参数
        "CNN_Filters",           # CNN滤波器数（逗号分隔，如"32,64"）
        "LSTM_Hidden",           # LSTM隐藏层大小
        # Transformer参数
        "Trans_DModel",          # Transformer的d_model
        "Trans_NHead",           # Transformer注意力头数
        "Trans_NLayers",         # Transformer编码器层数
        # 训练参数
        "Batch_Size",            # 批次大小
        "Learning_Rate",         # 学习率
        "Source_Epochs",         # 源域训练轮次
        "Target_Epochs",         # 目标域训练轮次
        "PINN_Alpha",            # PINN物理损失权重
        "Monotonic_Beta",        # 单调性损失权重
        "Adv_Lambda",            # 对抗损失权重
        "Loss_Function"          # 损失函数类型
    ]
    
    # 转换为DataFrame并保存
    df = pd.DataFrame(network_params_list, columns=columns)
    save_path = os.path.join(root_dir, "network_model_parameters.csv")
    df.to_csv(save_path, index=False, encoding="utf-8")
    print(f"💾 网络参数文件已保存: {save_path}")
    return save_path

def plot_error_curve(root_dir, target_key, method, eval_res):
    """（保持不变）绘制误差曲线"""
    sorted_idx = np.argsort(eval_res['true_rul'])[::-1]
    sorted_true = eval_res['true_rul'][sorted_idx]
    sorted_pred = eval_res['pred_rul'][sorted_idx]
    abs_error = np.abs(sorted_true - sorted_pred)
    
    model_name = method.replace('_', ' ').title()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    ax1.plot(sorted_true, label='True RUL', color='#2E86AB', linewidth=2.5)
    ax1.plot(sorted_pred, label='Predicted RUL', color='#A23B72', linewidth=2, alpha=0.8)
    ax1.set_ylabel('Normalized RUL', fontsize=12)
    ax1.set_title(f'{target_key} - {model_name}\nPrediction vs True RUL', fontsize=14)
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    ax2.plot(abs_error, color='#F18F01', linewidth=2, label='Absolute Error')
    ax2.set_xlabel('Sample Index (Sorted by True RUL Descending)', fontsize=12)
    ax2.set_ylabel('Absolute Error', fontsize=12)
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plot_path = os.path.join(root_dir, f"{target_key}_{method}_error_curve.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    return plot_path


# -------------------------- 6. 主实验流程 --------------------------
# （修改：集成网络参数收集与CSV保存）
def run_all_experiments():
    # 修改：获取带参数的结果文件夹路径
    results_root, param_str = create_results_root()
    performance_list = []
    # 新增：初始化网络参数列表（用于保存到CSV）
    network_params_list = []
    
    print(f"🚀 开始迁移学习与对比模型实验 | 结果目录: {results_root}\n")
    print(f"📋 关键网络参数配置: {param_str}\n")
    
    for exp_idx, exp in enumerate(experiments, 1):
        exp_name = exp['name']
        source_keys = exp['source_keys']
        target_key = exp['target_key']
        print(f"{'='*80}")
        print(f"实验 {exp_idx}/6: {exp_name}")
        print(f"{'='*80}")
        
        # 1. 数据准备（保持不变）
        print("\n1. 数据准备")
        source_X, source_y = merge_source_datasets(source_keys)
        source_X_tensor = prepare_tensors(source_X, device)
        source_y_tensor = torch.tensor(source_y, dtype=torch.float32).to(device)
        
        target_data = load_single_dataset(target_key)
        target_data = add_time_feature(target_data)
        target_X = target_data.iloc[:, :-1].values
        target_y = target_data.iloc[:, -1].values
        X_ft, y_ft, X_test, y_test = split_target_data(target_X, target_y)
        
        ft_X_tensor = prepare_tensors(X_ft, device)
        ft_y_tensor = torch.tensor(y_ft, dtype=torch.float32).to(device)
        test_X_tensor = prepare_tensors(X_test, device)
        test_y_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)
        
        # 2. 源域预训练模型（显式传入固定参数）
        input_size = source_X.shape[1]  # 动态获取输入特征数
        # 定义所有模型的固定结构参数（与模型默认值一致）
        fixed_params = {
            "base_hidden": 64,
            "tcn_chs": [32, 64],
            "gru_hid": 64,
            "gru_heads": 2,
            "cnn_filts": [32, 64],
            "lstm_hid": 64,
            "trans_d": 64,
            "trans_head": 2,
            "trans_layers": 1
        }
        
        # 显式传入hidden_size，确保参数可追溯
        source_model = BaseModel(input_size, hidden_size=fixed_params["base_hidden"]).to(device)
        source_model, pretrain_time = pretrain_source_model(
            source_model, source_X_tensor, source_y_tensor, device
        )
        
        # 3. 遍历所有方法（新增参数收集逻辑）
        for method_idx, method in enumerate(methods, 1):
            print(f"\n{'='*60}")
            print(f"方法 {method_idx}/{len(methods)}: {method.replace('_', ' ').title()}")
            print(f"{'='*60}")
            
            # 新增：初始化当前方法的参数字典（用于后续CSV保存）
            current_params = {
                "Experiment_Name": exp_name,
                "Model_Method": method,
                "Input_Size": input_size,
                # BaseModel参数
                "BaseModel_HiddenSize": fixed_params["base_hidden"],
                # TCN-BiGRU-Attention参数（格式化为字符串）
                "TCN_Channels": ",".join(map(str, fixed_params["tcn_chs"])),
                "GRU_Hidden": fixed_params["gru_hid"],
                "GRU_NumHeads": fixed_params["gru_heads"],
                # CNN-LSTM参数（格式化为字符串）
                "CNN_Filters": ",".join(map(str, fixed_params["cnn_filts"])),
                "LSTM_Hidden": fixed_params["lstm_hid"],
                # Transformer参数
                "Trans_DModel": fixed_params["trans_d"],
                "Trans_NHead": fixed_params["trans_head"],
                "Trans_NLayers": fixed_params["trans_layers"],
                # 训练参数（直接从全局变量读取）
                "Batch_Size": batch_size,
                "Learning_Rate": learning_rate,
                "Source_Epochs": source_epochs,
                "Target_Epochs": target_epochs,
                "PINN_Alpha": alpha,
                "Monotonic_Beta": beta,
                "Adv_Lambda": lambda_adv,
                "Loss_Function": loss_fn.__class__.__name__  # 获取损失函数类名（如MSELoss）
            }
            
            # 迁移学习方法（使用BaseModel）
            if method in ['fine_tuning', 'feature_extraction', 'adversarial']:
                current_params["Model_Type"] = "BaseModel"  # 标记模型类型
                target_model = BaseModel(input_size, hidden_size=fixed_params["base_hidden"]).to(device)
                
                if method == 'fine_tuning':
                    final_res, ft_time = train_fine_tuning(
                        source_model, target_model,
                        ft_X_tensor, ft_y_tensor,
                        test_X_tensor, test_y_tensor,
                        device
                    )
                    total_time = pretrain_time + ft_time
                    epochs_info = f"Source:{source_epochs},Target:{target_epochs}"
                
                elif method == 'feature_extraction':
                    final_res, ft_time = train_feature_extraction(
                        source_model, target_model,
                        ft_X_tensor, ft_y_tensor,
                        test_X_tensor, test_y_tensor,
                        device
                    )
                    total_time = pretrain_time + ft_time
                    epochs_info = f"Source:{source_epochs},Target:{target_epochs}"
                
                elif method == 'mmd_transfer':
                    final_res, ft_time = train_mmd_transfer(
                        source_model, target_model,
                        source_X_tensor, source_y_tensor,
                        ft_X_tensor, ft_y_tensor,
                        test_X_tensor, test_y_tensor,
                        device
                    )
                    total_time = pretrain_time + ft_time
                    epochs_info = f"Source:{source_epochs},Target:{target_epochs}"
                
                elif method == 'adversarial':
                    final_res, ft_time = train_adversarial(
                        source_model, target_model,
                        source_X_tensor, source_y_tensor,
                        ft_X_tensor, ft_y_tensor,
                        test_X_tensor, test_y_tensor,
                        device
                    )
                    total_time = pretrain_time + ft_time
                    epochs_info = f"Source:{source_epochs},Target:{target_epochs}"
            
            # 仅源域训练模型（根据方法确定模型类型）
            else:
                if method == 'tcn_bigru_attention':
                    model = TCNBiGRUAttention(
                        input_size,
                        tcn_channels=fixed_params["tcn_chs"],
                        gru_hidden=fixed_params["gru_hid"],
                        num_heads=fixed_params["gru_heads"]
                    ).to(device)
                    model_name = "TCN-BiGRU-Attention"
                    current_params["Model_Type"] = "TCNBiGRUAttention"  # 标记模型类型
                    final_res, total_time = train_source_only_model(
                        model, source_X_tensor, source_y_tensor,
                        test_X_tensor, test_y_tensor, device, model_name
                    )
                
                elif method == 'cnn_lstm':
                    model = CNNLSTM(
                        input_size,
                        cnn_filters=fixed_params["cnn_filts"],
                        lstm_hidden=fixed_params["lstm_hid"]
                    ).to(device)
                    model_name = "CNN-LSTM"
                    current_params["Model_Type"] = "CNNLSTM"  # 标记模型类型
                    final_res, total_time = train_source_only_model(
                        model, source_X_tensor, source_y_tensor,
                        test_X_tensor, test_y_tensor, device, model_name
                    )
                
                elif method == 'transformer':
                    model = TransformerModel(
                        input_size,
                        d_model=fixed_params["trans_d"],
                        nhead=fixed_params["trans_head"],
                        num_layers=fixed_params["trans_layers"]
                    ).to(device)
                    model_name = "Transformer"
                    current_params["Model_Type"] = "TransformerModel"  # 标记模型类型
                    final_res, total_time = train_source_only_model(
                        model, source_X_tensor, source_y_tensor,
                        test_X_tensor, test_y_tensor, device, model_name
                    )
                
                epochs_info = f"Source:{source_epochs}"
            
            # 新增：将当前方法的参数添加到列表
            network_params_list.append(current_params)
            
            # 4. 结果保存（原逻辑不变）
            print(f"\n📊 方法 {method} 结果 | MAE: {final_res['mae']:.4f} | RMSE: {final_res['rmse']:.4f} | R2: {final_res['r2']:.4f}")
            
            pred_path = save_prediction_csv(results_root, target_key, method, final_res)
            print(f"💾 预测结果: {pred_path}")
            
            plot_path = plot_error_curve(results_root, target_key, method, final_res)
            print(f"📈 误差曲线: {plot_path}")
            
            performance = {
                'Dataset': target_key,
                'Model': method.replace('_', ' ').title(),
                'MAE': round(final_res['mae'], 4),
                'RMSE': round(final_res['rmse'], 4),
                'R2': round(final_res['r2'], 4),
                'Train_Time(s)': round(total_time, 2),
                'Batch_Size': batch_size,
                'Epochs': epochs_info,
                'Device': str(device)
            }
            performance_list.append(performance)
            
            if exp_idx == 1 and method_idx == 1:
                save_performance_csv(results_root, performance_list, is_init=True)
            else:
                save_performance_csv(results_root, performance_list[-1:], is_init=False)
        
        print(f"\n{'='*80}\n")
    
    # 新增：所有实验结束后，保存网络模型参数CSV
    save_network_params_csv(results_root, network_params_list)
    
    print(f"🎉 所有实验完成！")
    print(f"📁 结果总目录: {results_root}")
    print(f"📄 性能汇总表: {os.path.join(results_root, 'all_models_performance.csv')}")
    print(f"📄 网络参数表: {os.path.join(results_root, 'network_model_parameters.csv')}")


# -------------------------- 7. 启动实验 --------------------------
if __name__ == "__main__":
    start_time = time.time()
    run_all_experiments()
    total_time = time.time() - start_time
    print(f"\n⏱️  总耗时: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")
