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

# ========================== 1. 核心参数配置 ==========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 数据集路径
datasets = {
    'C1': 'filtered_data_normalized-C1.csv',
    'C4': 'filtered_data_normalized-C4.csv',
    'C6': 'filtered_data_normalized-C6.csv',
    'RC1': 'filtered_data_normalized-ROBOT-C1.csv',
    'RC2': 'filtered_data_normalized-ROBOT-C2.csv',
    'RC3': 'filtered_data_normalized-ROBOT-C3.csv'
}

# 实验配置：6 组源→目标
experiments = [
    {'name': 'C1+C4+C6→RC1', 'source_keys': ['C1', 'C4', 'C6'], 'target_key': 'RC1'},
    {'name': 'C1+C4+C6→RC2', 'source_keys': ['C1', 'C4', 'C6'], 'target_key': 'RC2'},
    {'name': 'C1+C4+C6→RC3', 'source_keys': ['C1', 'C4', 'C6'], 'target_key': 'RC3'},
]

# 训练轮数（可按需修改）
source_epochs = 120   # 源域预训练轮数
target_epochs = 120   # 目标域微调轮数

# 其他训练参数
batch_size = 64
learning_rate = 0.001
alpha = 0.01      # PINN 物理损失权重
beta = 0.01       # 单调性损失权重
loss_fn = nn.MSELoss()

# 早停配置（只作用于 fine_tuning 阶段）
EARLY_STOPPING = True
PATIENCE = 10        # 连续多少次评估无提升就停止
MIN_DELTA = 0.0      # RMSE 至少提升多少才算“有进步”
EVAL_INTERVAL = 5    # 每多少个 epoch 在测试集上评估一次


# ========================== 2. 数据处理工具函数 ==========================
def load_single_dataset(dataset_key):
    """根据 key 读单个数据集，默认最后一列为 RUL 标签"""
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
    """若无 Time 列，则插入一个归一化时间特征 [0,1]"""
    if 'Time' not in data.columns:
        data.insert(0, 'Time', np.linspace(0, 1, len(data)))
    return data


def merge_source_datasets(source_keys):
    """按源域列表合并数据集"""
    merged_X, merged_y = None, None
    for key in source_keys:
        data = load_single_dataset(key)
        data = add_time_feature(data)
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
        if merged_X is None:
            merged_X, merged_y = X, y
        else:
            merged_X = np.concatenate([merged_X, X], axis=0)
            merged_y = np.concatenate([merged_y, y], axis=0)
    print(f"📊 合并源域 {source_keys} | 总样本数: {len(merged_X)} | 特征数: {merged_X.shape[1]}")
    return merged_X, merged_y


def split_target_data(X_target, y_target):
    """
    目标域划分：
      - 测试集：每隔 5 个样本取 1 个（约 20%）
      - 微调集：在剩余样本中每隔 8 个取 1 个
    """
    total_indices = np.arange(len(X_target))

    # 测试集
    test_indices = total_indices[::5]
    X_test = X_target[test_indices]
    y_test = y_target[test_indices]

    # 微调集
    remaining_indices = np.setdiff1d(total_indices, test_indices)
    if len(remaining_indices) < 8:
        ft_indices = remaining_indices
    else:
        ft_indices = remaining_indices[::8]

    X_ft = X_target[ft_indices]
    y_ft = y_target[ft_indices]

    print(f"🎯 目标域采样 | 微调集: {len(X_ft)} 样本 | 测试集: {len(X_test)} 样本")
    return X_ft, y_ft, X_test, y_test


def prepare_tensors(data, device):
    """转换为时序模型标准格式：[样本数, seq_len=1, 特征数]"""
    tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(1)
    return tensor.to(device)


# ========================== 3. BLCAP 模型定义（BaseModel） ==========================
class ChannelAttention(nn.Module):
    """通道注意力机制：对 BiLSTM 输出的通道进行加权"""
    def __init__(self, hidden_size, reduction_ratio=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size // reduction_ratio, hidden_size, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [batch, seq_len, hidden_size]
        b, l, c = x.size()
        y = self.avg_pool(x.permute(0, 2, 1)).view(b, c)  # 通道全局池化
        y = self.fc(y).view(b, 1, c)                      # 生成通道权重
        return x * y.expand_as(x)                         # 施加注意力权重


class BaseModel(nn.Module):
    """
    BLCAP 主干：
      - BiLSTM 提取时序特征
      - ChannelAttention 建模不同隐通道的重要性
      - 全连接回归 RUL（0~1）
      - physics_loss + monotonicity_loss 物理/单调约束
    """
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.hidden_size = hidden_size
        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=True,
            batch_first=True
        )
        self.channel_attn = ChannelAttention(hidden_size * 2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.time_idx = 0  # 假设第 0 维是 Time 特征

    def forward(self, x):
        # x: [batch, seq_len, features]
        x, _ = self.bilstm(x)           # [batch, seq_len, hidden_size*2]
        x = self.channel_attn(x)
        return self.fc(x[:, -1, :])     # 取最后时间步输出

    def physics_loss(self, y_pred, x):
        """
        PINN 物理约束：RUL 随时间递减，理想关系：y(t) <= 1 - t
        用 ReLU 强制对违反约束的部分进行惩罚。
        """
        t = x[:, :, self.time_idx].squeeze()  # [batch] 或 [batch, seq_len]
        return torch.mean(torch.relu(y_pred.squeeze() - (1 - t)))

    def monotonicity_loss(self, y_pred, x):
        """
        单调性约束：随时间 t 增大，RUL 不能上升。
        将样本按 t 排序，约束相邻差分 y_{i+1} - y_i <= 0。
        """
        t = x[:, :, self.time_idx].squeeze()
        sorted_idx = torch.argsort(t)
        sorted_pred = y_pred[sorted_idx]
        diffs = sorted_pred[1:] - sorted_pred[:-1]
        return torch.mean(torch.relu(diffs))


# ========================== 4. 评估与训练函数 ==========================
def evaluate_model(model, X_tensor, y_tensor, return_time=False):
    """在给定数据集上评估模型，可选返回一次完整推理时间"""
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        y_pred = model(X_tensor).cpu().numpy().flatten()
        y_true = y_tensor.cpu().numpy().flatten()
    infer_time = time.time() - start_time

    metrics = {
        'true_rul': y_true,
        'pred_rul': y_pred,
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred)
    }

    if return_time:
        return metrics, infer_time
    else:
        return metrics


def pretrain_source_model(model, X_train, y_train, device, epochs):
    """源域预训练（BLCAP 主干），不加早停，直接跑满 epochs"""
    dataset = TensorDataset(X_train, y_train.unsqueeze(1))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            y_pred = model(batch_x)
            data_loss = loss_fn(y_pred, batch_y)
            pde_loss = model.physics_loss(y_pred, batch_x)
            mono_loss = model.monotonicity_loss(y_pred, batch_x)
            loss = data_loss + alpha * pde_loss + beta * mono_loss

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)

        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / len(loader.dataset)
            print(f"  源域预训练 Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f}")

    pretrain_time = time.time() - start_time
    print(f"📌 源域预训练完成 | 耗时: {pretrain_time:.2f} 秒")
    return model, pretrain_time


def train_fine_tuning(source_model, target_model,
                      X_ft, y_ft, X_test, y_test,
                      device, target_epochs,
                      early_stopping=EARLY_STOPPING,
                      patience=PATIENCE,
                      min_delta=MIN_DELTA,
                      eval_interval=EVAL_INTERVAL):
    """
    BLCAP + fine_tuning：
      - 先加载源域预训练权重
      - 在目标域微调（带物理/单调损失）
      - 按目标域 Test RMSE 早停
    返回：(最终评估结果, 微调训练耗时, 推理耗时)
    """
    print(f"\n📌 微调迁移学习训练 (BLCAP + Fine-Tuning)，目标域 epochs={target_epochs}")
    target_model.load_state_dict(source_model.state_dict())

    dataset = TensorDataset(X_ft, y_ft.unsqueeze(1))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.AdamW(target_model.parameters(), lr=learning_rate / 10, weight_decay=1e-4)

    start_time = time.time()
    best_rmse = float('inf')
    best_state = None
    epochs_no_improve = 0

    for epoch in range(target_epochs):
        target_model.train()
        total_loss = 0.0

        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            y_pred = target_model(batch_x)
            data_loss = loss_fn(y_pred, batch_y)
            pde_loss = target_model.physics_loss(y_pred, batch_x)
            mono_loss = target_model.monotonicity_loss(y_pred, batch_x)
            loss = data_loss + alpha * pde_loss + beta * mono_loss

            loss.backward()
            nn.utils.clip_grad_norm_(target_model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)

        # 定期在目标域测试集上评估 RMSE，用于早停
        if (epoch + 1) % eval_interval == 0 or epoch == target_epochs - 1:
            avg_loss = total_loss / len(loader.dataset)
            eval_res = evaluate_model(target_model, X_test, y_test.unsqueeze(1))
            rmse = eval_res['rmse']
            print(f"  Epoch [{epoch+1}/{target_epochs}] | Loss: {avg_loss:.4f} | Test RMSE: {rmse:.4f}")

            if rmse < best_rmse - min_delta:
                best_rmse = rmse
                epochs_no_improve = 0
                best_state = {k: v.cpu().clone() for k, v in target_model.state_dict().items()}
            else:
                epochs_no_improve += 1
                if early_stopping and epochs_no_improve >= patience:
                    print(f"⏹️  早停触发：连续 {epochs_no_improve} 次评估 RMSE 未提升，停止微调。")
                    break

    # 恢复到 RMSE 最优的参数
    if best_state is not None:
        target_model.load_state_dict(best_state)

    ft_time = time.time() - start_time
    final_res, infer_time = evaluate_model(target_model, X_test, y_test.unsqueeze(1), return_time=True)
    return final_res, ft_time, infer_time


# ========================== 5. 结果保存与可视化 ==========================
def create_results_root(source_epochs, target_epochs):
    """生成结果根目录，包含关键参数+时间戳"""
    base_hidden = 64
    param_str = f"BLCAP_hid{base_hidden}_SrcEp{source_epochs}_TgtEp{target_epochs}"
    time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    root_dir = f"BLCAP_finetune_{param_str}_{time_str}"
    os.makedirs(root_dir, exist_ok=True)
    return root_dir, param_str


def save_prediction_csv(root_dir, target_key, eval_res,
                        source_epochs, target_epochs):
    """保存预测结果"""
    sorted_idx = np.argsort(eval_res['true_rul'])[::-1]
    sorted_true = eval_res['true_rul'][sorted_idx]
    sorted_pred = eval_res['pred_rul'][sorted_idx]
    abs_error = np.abs(sorted_true - sorted_pred)

    df = pd.DataFrame({
        'Dataset': [target_key] * len(sorted_true),
        'Model': ['BLCAP+FineTuning'] * len(sorted_true),
        'Source_Epochs': [source_epochs] * len(sorted_true),
        'Target_Epochs': [target_epochs] * len(sorted_true),
        'True_RUL': sorted_true,
        'Predicted_RUL': sorted_pred,
        'Absolute_Error': abs_error
    })

    save_path = os.path.join(
        root_dir,
        f"{target_key}_BLCAP_finetune_S{source_epochs}_T{target_epochs}_predictions.csv"
    )
    df.to_csv(save_path, index=False)
    return save_path


def save_performance_csv(root_dir, performance_list, is_init=False):
    """保存性能汇总表"""
    df = pd.DataFrame(performance_list)
    save_path = os.path.join(root_dir, "BLCAP_finetune_performance.csv")
    if is_init:
        df.to_csv(save_path, index=False, mode='w')
    else:
        df.to_csv(save_path, index=False, mode='a', header=False)
    return save_path


def save_network_params_csv(root_dir, network_params_list):
    """保存网络结构与训练超参数"""
    columns = [
        "Experiment_Name",
        "Model_Name",
        "Model_Type",
        "Input_Size",
        "Base_HiddenSize",
        "Batch_Size",
        "Learning_Rate",
        "Source_Epochs",
        "Target_Epochs",
        "PINN_Alpha",
        "Monotonic_Beta",
        "Loss_Function"
    ]
    df = pd.DataFrame(network_params_list, columns=columns)
    save_path = os.path.join(root_dir, "BLCAP_network_parameters.csv")
    df.to_csv(save_path, index=False, encoding="utf-8")
    print(f"💾 网络参数文件已保存: {save_path}")
    return save_path


def plot_error_curve(root_dir, target_key, eval_res,
                     source_epochs, target_epochs):
    """绘制误差曲线"""
    sorted_idx = np.argsort(eval_res['true_rul'])[::-1]
    sorted_true = eval_res['true_rul'][sorted_idx]
    sorted_pred = eval_res['pred_rul'][sorted_idx]
    abs_error = np.abs(sorted_true - sorted_pred)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(sorted_true, label='True RUL', linewidth=2)
    ax1.plot(sorted_pred, label='Predicted RUL', linewidth=2, alpha=0.8)
    ax1.set_ylabel('Normalized RUL')
    ax1.set_title(f'{target_key} - BLCAP+FineTuning\n'
                  f'Source_Epochs={source_epochs}, Target_Epochs={target_epochs}')
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(abs_error, label='Absolute Error', linewidth=2)
    ax2.set_xlabel('Sample Index (sorted by True RUL desc)')
    ax2.set_ylabel('Absolute Error')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plot_path = os.path.join(
        root_dir,
        f"{target_key}_BLCAP_finetune_S{source_epochs}_T{target_epochs}_error_curve.png"
    )
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    return plot_path


# ========================== 6. 主实验流程：只跑 BLCAP+FineTuning ==========================
def run_blcap_finetune_experiments(source_epochs, target_epochs):
    results_root, param_str = create_results_root(source_epochs, target_epochs)
    performance_list = []
    network_params_list = []

    print(f"🚀 开始 BLCAP + Fine-Tuning 实验 | 源域 epochs={source_epochs}, 目标域 epochs={target_epochs}")
    print(f"📁 结果目录: {results_root}")
    print(f"📋 配置摘要: {param_str}\n")

    for exp_idx, exp in enumerate(experiments, 1):
        exp_name = exp['name']
        source_keys = exp['source_keys']
        target_key = exp['target_key']
        print(f"{'=' * 80}")
        print(f"实验 {exp_idx}/6: {exp_name}")
        print(f"{'=' * 80}")

        # 1. 数据准备
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

        input_size = source_X.shape[1]

        # 2. 源域 BLCAP 预训练
        print("\n2. 源域 BLCAP 预训练")
        source_model = BaseModel(input_size, hidden_size=64).to(device)
        source_model, pretrain_time = pretrain_source_model(
            source_model, source_X_tensor, source_y_tensor, device, source_epochs
        )

        # 3. 目标域 BLCAP + Fine-Tuning（带早停）
        print("\n3. 目标域 BLCAP + Fine-Tuning")
        target_model = BaseModel(input_size, hidden_size=64).to(device)
        final_res, ft_time, infer_time = train_fine_tuning(
            source_model, target_model,
            ft_X_tensor, ft_y_tensor,
            test_X_tensor, test_y_tensor,
            device, target_epochs
        )

        total_train_time = pretrain_time + ft_time

        # 4. 保存结果
        print(f"\n📊 实验 {exp_name} 结果 | MAE: {final_res['mae']:.4f} | "
              f"RMSE: {final_res['rmse']:.4f} | R2: {final_res['r2']:.4f}")
        print(f"⏱️  预训练: {pretrain_time:.2f}s | 微调: {ft_time:.2f}s | 总训练: {total_train_time:.2f}s "
              f"| 推理: {infer_time:.4f}s")

        pred_path = save_prediction_csv(
            results_root, target_key, final_res,
            source_epochs, target_epochs
        )
        print(f"💾 预测结果 CSV: {pred_path}")

        plot_path = plot_error_curve(
            results_root, target_key, final_res,
            source_epochs, target_epochs
        )
        print(f"📈 误差曲线 PNG: {plot_path}")

        # 性能记录
        performance = {
            'Experiment': exp_name,
            'Target_Dataset': target_key,
            'Model': 'BLCAP+FineTuning',
            'Source_Epochs': source_epochs,
            'Target_Epochs': target_epochs,
            'MAE': round(final_res['mae'], 4),
            'RMSE': round(final_res['rmse'], 4),
            'R2': round(final_res['r2'], 4),
            'Pretrain_Time(s)': round(pretrain_time, 2),
            'FT_Time(s)': round(ft_time, 2),
            'Train_Time_Total(s)': round(total_train_time, 2),
            'Infer_Time(s)': round(infer_time, 4),
            'Batch_Size': batch_size,
            'Device': str(device)
        }
        performance_list.append(performance)

        # 网络参数记录
        net_params = [
            exp_name,
            'BLCAP+FineTuning',
            'BaseModel(BLCAP)',
            input_size,
            64,
            batch_size,
            learning_rate,
            source_epochs,
            target_epochs,
            alpha,
            beta,
            loss_fn.__class__.__name__
        ]
        network_params_list.append(net_params)

        # 写入性能 CSV（追加）
        if exp_idx == 1:
            save_performance_csv(results_root, performance_list, is_init=True)
        else:
            save_performance_csv(results_root, performance_list[-1:], is_init=False)

        print(f"\n{'=' * 80}\n")

    # 写入网络参数 CSV
    save_network_params_csv(results_root, network_params_list)

    print(f"🎉 BLCAP + Fine-Tuning 全部 6 组实验完成！")
    print(f"📁 结果目录: {results_root}")
    print(f"📄 性能汇总表: {os.path.join(results_root, 'BLCAP_finetune_performance.csv')}")
    print(f"📄 网络参数表: {os.path.join(results_root, 'BLCAP_network_parameters.csv')}")
    return results_root


# ========================== 7. 主入口 ==========================
if __name__ == "__main__":
    start_time = time.time()
    results_dir = run_blcap_finetune_experiments(source_epochs, target_epochs)
    total_time = time.time() - start_time
    print(f"\n⏱️  所有 BLCAP+FineTuning 实验总耗时: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")
    print(f"📁  最终结果目录: {results_dir}")

