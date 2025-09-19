import os
import sys
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

# 复用数据与IO函数
from bayesian_timeseries import (
    DATA_DIRS,
    build_dataset,
    get_output_path,
    get_available_materials,
    train_test_split_indices,
)


# 切到脚本所在目录，确保相对路径正确
script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
os.chdir(script_dir)


def seed_everything(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 材料参数（沿用 main.py 的定义）
materials_params = {
    'AISI316L': {
        'G': 67.3,
        'tau_f_G': 430.4,
        'gamma_f': 0.279,
        'b0': -0.092,
        'c0': -0.419,
    },
    'GH4169': {
        'G': 67.0,
        'tau_f_G': 1091.6,
        'gamma_f': 4.46,
        'b0': -0.07,
        'c0': -0.77,
    },
    'TC4': {
        'G': 43.2,
        'tau_f_G': 716.9,
        'gamma_f': 2.24,
        'b0': -0.06,
        'c0': -0.8,
    },
    'CuZn37': {
        'G': 49.6,
        'tau_f_G': 356.1,
        'gamma_f': 0.068,
        'b0': -0.0816,
        'c0': -0.3298,
    },
    'Q235B1': {
        'G': 81.4,
        'tau_f_G': 308.9,
        'gamma_f': 0.9751,
        'b0': -0.0702,
        'c0': -0.6723,
    },
    'Q235B2': {
        'G': 76.3,
        'tau_f_G': 299.9,
        'gamma_f': 0.035,
        'b0': -0.3555,
        'c0': -0.3181,
    },
}


def fs_loss(Np_pred: torch.Tensor, FP_true: torch.Tensor, material_name: str, eps: float = 1e-6) -> torch.Tensor:
    """
    FS 准则物理损失（与 main.py 一致的形式）：
    - 使用材料参数 (G, tau_f_G, gamma_f, b0, c0)
    - 预测 FP_pred = (tau_f'/G)*(2*Np)^b0 + gamma_f'*(2*Np)^c0
    - 在 log10 空间以相对误差的 Huber 形式进行度量
    """
    params = materials_params[material_name]
    tau_f_prime = torch.tensor(params['tau_f_G'], device=Np_pred.device, dtype=Np_pred.dtype)
    gamma_f_prime = torch.tensor(params['gamma_f'], device=Np_pred.device, dtype=Np_pred.dtype)
    b0 = torch.tensor(params['b0'], device=Np_pred.device, dtype=Np_pred.dtype)
    c0 = torch.tensor(params['c0'], device=Np_pred.device, dtype=Np_pred.dtype)
    G = torch.tensor(params['G'], device=Np_pred.device, dtype=Np_pred.dtype)

    # 避免 0 或负数，且避免过高值造成数值不稳
    Np_safe = torch.clamp(Np_pred, min=eps, max=1e5)

    # 预测 FP
    term1 = (tau_f_prime / G) * torch.pow(2.0 * Np_safe, b0)
    term2 = gamma_f_prime * torch.pow(2.0 * Np_safe, c0)
    FP_pred = term1 + term2

    # clamp 真值，避免 log10 NaN
    FP_pos = torch.clamp(FP_true, min=eps)

    log_FP_pred = torch.log10(FP_pred + eps)
    log_FP = torch.log10(FP_pos + eps)

    # 相对误差的 Huber
    relative_error = torch.abs(log_FP_pred - log_FP) / (torch.abs(log_FP) + eps)
    delta = 0.1
    huber_mask = relative_error < delta
    loss = torch.where(huber_mask, 0.5 * relative_error ** 2, delta * relative_error - 0.5 * delta ** 2)
    return loss.mean()


def build_mlp(input_dim: int, hidden_dims: list[int], dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(nn.ReLU())
        if dropout > 0.0:
            layers.append(nn.Dropout(p=float(dropout)))
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, 1))
    return nn.Sequential(*layers)


class MultiTaskMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float):
        super().__init__()
        backbone_layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            backbone_layers.append(nn.Linear(prev_dim, hidden_dim))
            backbone_layers.append(nn.ReLU())
            if dropout > 0.0:
                backbone_layers.append(nn.Dropout(p=float(dropout)))
            prev_dim = hidden_dim
        self.backbone = nn.Sequential(*backbone_layers)
        self.life_head = nn.Linear(prev_dim, 1)
        self.mech_head = nn.Linear(prev_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        life = self.life_head(h).squeeze(-1)
        mech_prob = torch.sigmoid(self.mech_head(h)).squeeze(-1)
        return life, mech_prob


def soft_binary_cross_entropy(pred_prob: torch.Tensor, target_soft: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred_clamped = torch.clamp(pred_prob, min=eps, max=1.0 - eps)
    return F.binary_cross_entropy(pred_clamped, target_soft)


def generate_soft_mechanism_label(material: str, gamma_a: float, epsilon_a: float) -> float:
    ratio = gamma_a / epsilon_a if epsilon_a != 0 else 0.0
    if material == 'TC4':
        return 1.0 if ratio < 1.4 else (0.6 if ratio < 1.6 else 0.0)
    if material == 'GH4169':
        return 1.0 if ratio < 1.35 else (0.5 if ratio < 1.55 else 0.0)
    if material == 'CuZn37':
        if ratio < 1.0:
            return 0.0
        elif ratio < 1.55:
            return 0.5
        elif ratio <= 1.6:
            return 0.6
        else:
            return 1.0
    return -1.0


def train_mflp_pinn(
    X_train: np.ndarray,
    y_train_log10: np.ndarray,
    *,
    hidden_dims: list[int],
    dropout: float,
    lr: float,
    weight_decay: float,
    epochs: int,
    batch_size: int,
    upper_cycle_limit: float,
    lambda_nonneg: float,
    lambda_upper: float,
    lambda_fs: float,
    lambda_mech: float,
    fp_train: np.ndarray,
    mech_labels_train: np.ndarray,
    material_name: str,
    device: torch.device,
    grad_clip_norm: float | None = None,
):
    model = MultiTaskMLP(X_train.shape[1], hidden_dims, dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    X_tr = torch.from_numpy(X_train.astype(np.float32)).to(device)
    y_tr_log = torch.from_numpy(y_train_log10.astype(np.float32)).to(device)
    FP_tr = torch.from_numpy(fp_train.astype(np.float32)).to(device)
    mech_tr = torch.from_numpy(mech_labels_train.astype(np.float32)).to(device)

    num_samples = X_tr.shape[0]
    num_batches = max(1, int(math.ceil(num_samples / max(1, batch_size))))

    model.train()
    for epoch in range(int(epochs)):
        perm = torch.randperm(num_samples, device=device)
        epoch_loss = 0.0
        for b in range(num_batches):
            idx = perm[b * batch_size : (b + 1) * batch_size]
            x_b = X_tr.index_select(0, idx)
            y_b_log = y_tr_log.index_select(0, idx)

            # 模型输出：寿命与机制概率
            Np_pred, mech_prob = model(x_b)

            # 主损失：在 log10 空间与真实值进行 MSE
            pred_log = torch.log10(torch.clamp(Np_pred, min=1e-12))
            loss_mse = F.mse_loss(pred_log, y_b_log)

            # 物理约束：非负与上界（例如 1e7 次循环）
            loss_phys_low = F.relu(-Np_pred).mean() * float(lambda_nonneg)
            loss_phys_high = F.relu(Np_pred - float(upper_cycle_limit)).mean() * float(lambda_upper)

            # FS 物理损失（与 main.py 一致）
            FP_b = FP_tr.index_select(0, idx)
            loss_fs_val = fs_loss(Np_pred, FP_b, material_name) * float(lambda_fs)

            # 机制分类软标签损失（仅对有效标签计算）
            mech_b = mech_tr.index_select(0, idx)
            valid_mask = mech_b >= 0.0
            if torch.any(valid_mask):
                loss_mech = soft_binary_cross_entropy(mech_prob[valid_mask], mech_b[valid_mask]) * float(lambda_mech)
            else:
                loss_mech = torch.tensor(0.0, device=device)

            loss = loss_mse + loss_phys_low + loss_phys_high + loss_fs_val + loss_mech

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip_norm is not None and grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
            optimizer.step()

            epoch_loss += float(loss.detach().cpu())

        # 可选打印
        if (epoch + 1) % max(1, int(epochs // 10)) == 0:
            print(f'Epoch {epoch + 1}/{epochs} - loss: {epoch_loss / num_batches:.6f}')

    model.eval()
    return model


def predict_cycles(model: nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        X_t = torch.from_numpy(X.astype(np.float32)).to(device)
        if isinstance(model, MultiTaskMLP):
            Np_pred, _ = model(X_t)
        else:
            Np_pred = model(X_t).squeeze(-1)
        return Np_pred.detach().cpu().numpy().astype(np.float64)


def predict_outputs(model: nn.Module, X: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        X_t = torch.from_numpy(X.astype(np.float32)).to(device)
        if isinstance(model, MultiTaskMLP):
            Np_pred, mech_prob = model(X_t)
        else:
            Np_pred = model(X_t).squeeze(-1)
            mech_prob = torch.zeros_like(Np_pred)
        return (
            Np_pred.detach().cpu().numpy().astype(np.float64),
            mech_prob.detach().cpu().numpy().astype(np.float64),
        )


def evaluate_and_save_split(
    material: str,
    dataset: dict,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    Np_tr: np.ndarray,
    Np_te: np.ndarray,
    Mech_tr: np.ndarray,
    Mech_te: np.ndarray,
):
    meta = dataset['meta']

    Nf_tr = np.array([meta[i]['Nf'] for i in train_idx], dtype=np.float64)
    Nf_te = np.array([meta[i]['Nf'] for i in test_idx], dtype=np.float64)

    # 评估（log 空间误差 + 2倍误差内比例）
    mu_tr_log = np.log10(np.clip(Np_tr, 1e-12, None))
    mu_te_log = np.log10(np.clip(Np_te, 1e-12, None))
    log_err_tr = np.abs(mu_tr_log - np.log10(Nf_tr))
    log_err_te = np.abs(mu_te_log - np.log10(Nf_te))
    print(f'训练集: n={train_idx.size}, 平均对数误差={float(np.mean(log_err_tr)):.4f}, 2倍误差内比例={float(np.mean(log_err_tr <= np.log10(2.0)) * 100.0):.2f}%')
    print(f'测试集: n={test_idx.size}, 平均对数误差={float(np.mean(log_err_te)):.4f}, 2倍误差内比例={float(np.mean(log_err_te <= np.log10(2.0)) * 100.0):.2f}%')

    # 保存 CSV（无不确定性，lo/hi 与 mean 相同）
    rows = []
    for k, j in enumerate(train_idx):
        m = meta[j]
        rows.append({
            'set': 'train',
            'file': m['file'],
            'epsilon_a': m['epsilon_a'],
            'gamma_a': m['gamma_a'],
            'FP': m['FP'],
            'Nf_true': m['Nf'],
            'Np_pred_mean': float(Np_tr[k]),
            'Np_pred_lo': float(Np_tr[k]),
            'Np_pred_hi': float(Np_tr[k]),
            'mech_prob': float(Mech_tr[k]),
            'log_error': float(log_err_tr[k])
        })
    for k, j in enumerate(test_idx):
        m = meta[j]
        rows.append({
            'set': 'test',
            'file': m['file'],
            'epsilon_a': m['epsilon_a'],
            'gamma_a': m['gamma_a'],
            'FP': m['FP'],
            'Nf_true': m['Nf'],
            'Np_pred_mean': float(Np_te[k]),
            'Np_pred_lo': float(Np_te[k]),
            'Np_pred_hi': float(Np_te[k]),
            'mech_prob': float(Mech_te[k]),
            'log_error': float(log_err_te[k])
        })

    import pandas as pd  # 延迟导入，避免未使用时的依赖
    df_out = pd.DataFrame(rows)
    csv_path = get_output_path(material, f'{material}_mflp_pinn_results_split.csv')
    df_out.to_csv(csv_path, index=False)
    print(f'分割结果已保存: {csv_path}')

    # 绘图（训练/测试分色）
    plt.figure(figsize=(8, 8))
    plt.scatter(Nf_tr, Np_tr, alpha=0.75, label='Train', c='#1f77b4', s=60)
    plt.scatter(Nf_te, Np_te, alpha=0.95, label='Test', c='#d62728', s=120, edgecolors='k', linewidths=0.5, zorder=3)
    mn = min(float(np.min(np.concatenate([Nf_tr, Nf_te]))), float(np.min(np.concatenate([Np_tr, Np_te]))))
    mx = max(float(np.max(np.concatenate([Nf_tr, Nf_te]))), float(np.max(np.concatenate([Np_tr, Np_te]))))
    plt.plot([mn, mx], [mn, mx], 'r--', label='Perfect')
    plt.plot([mn, mx], [mn / 2.0, mx / 2.0], 'k:', label='Factor of 2')
    plt.plot([mn, mx], [mn * 2.0, mx * 2.0], 'k:')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('True Life (Nf)')
    plt.ylabel('Predicted Life (Np)')
    plt.title(f'{material} MFLP-PINN Predictions (Train/Test)')
    plt.legend()
    plt.grid(True)
    plot_path = get_output_path(material, f'{material}_mflp_pinn_scatter_split.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'分色散点图已保存: {plot_path}')


def evaluate_and_save_loo(
    material: str,
    dataset: dict,
    Np_pred_all: np.ndarray,
    Mech_prob_all: np.ndarray,
):
    meta = dataset['meta']
    Nf_true = np.array([m['Nf'] for m in meta], dtype=np.float64)
    mu_log = np.log10(np.clip(Np_pred_all, 1e-12, None))
    log_err = np.abs(mu_log - np.log10(Nf_true))
    print(f'样本数: {Nf_true.size}, 平均对数误差: {float(np.mean(log_err)):.4f}, 2倍误差内比例: {float(np.mean(log_err <= np.log10(2.0)) * 100.0):.2f}%')

    # 保存 CSV
    rows = []
    for i, m in enumerate(meta):
        rows.append({
            'file': m['file'],
            'epsilon_a': m['epsilon_a'],
            'gamma_a': m['gamma_a'],
            'FP': m['FP'],
            'Nf_true': m['Nf'],
            'Np_pred_mean': float(Np_pred_all[i]),
            'Np_pred_lo': float(Np_pred_all[i]),
            'Np_pred_hi': float(Np_pred_all[i]),
            'mech_prob': float(Mech_prob_all[i]),
            'log_error': float(log_err[i])
        })
    import pandas as pd
    df_out = pd.DataFrame(rows)
    csv_path = get_output_path(material, f'{material}_mflp_pinn_results_loo.csv')
    df_out.to_csv(csv_path, index=False)
    print(f'结果已保存: {csv_path}')

    # 绘图
    plt.figure(figsize=(8, 8))
    plt.scatter(Nf_true, Np_pred_all, alpha=0.7, label='Predicted')
    mn = min(float(np.min(Nf_true)), float(np.min(Np_pred_all)))
    mx = max(float(np.max(Nf_true)), float(np.max(Np_pred_all)))
    plt.plot([mn, mx], [mn, mx], 'r--', label='Perfect')
    plt.plot([mn, mx], [mn / 2.0, mx / 2.0], 'k:', label='Factor of 2')
    plt.plot([mn, mx], [mn * 2.0, mx * 2.0], 'k:')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('True Life (Nf)')
    plt.ylabel('Predicted Life (Np)')
    plt.title(f'{material} MFLP-PINN Predictions (LOO)')
    plt.legend()
    plt.grid(True)
    plot_path = get_output_path(material, f'{material}_mflp_pinn_scatter_loo.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'散点图已保存: {plot_path}')


def run_material_split(
    material: str,
    *,
    include_fp: bool,
    hidden_dims: list[int],
    dropout: float,
    lr: float,
    weight_decay: float,
    epochs: int,
    batch_size: int,
    upper_cycle_limit: float,
    lambda_nonneg: float,
    lambda_upper: float,
    lambda_fs: float,
    lambda_mech: float,
    test_ratio: float,
    seed: int,
    device: torch.device,
):
    print(f'开始处理材料: {material} (MFLP-PINN + 时间序列特征，Train/Test 分色)')
    dataset = build_dataset(material, include_fp=include_fp)
    X = dataset['X']
    y_log = dataset['y']  # log10(Nf)
    mu = dataset['mu']
    sigma = dataset['sigma']

    # 标准化
    sigma_safe = np.where(sigma > 0, sigma, 1.0)
    Xz = (X - mu) / sigma_safe

    # 划分
    n = Xz.shape[0]
    train_idx, test_idx = train_test_split_indices(n, test_ratio=float(test_ratio), seed=int(seed))

    # 取 FP 真值
    fp_all = np.array([m['FP'] for m in dataset['meta']], dtype=np.float64)
    # 机制软标签
    mech_all = np.array([
        generate_soft_mechanism_label(material, float(m['gamma_a']), float(m['epsilon_a']))
        for m in dataset['meta']
    ], dtype=np.float64)

    # 训练
    model = train_mflp_pinn(
        X_train=Xz[train_idx],
        y_train_log10=y_log[train_idx],
        hidden_dims=hidden_dims,
        dropout=dropout,
        lr=lr,
        weight_decay=weight_decay,
        epochs=epochs,
        batch_size=batch_size,
        upper_cycle_limit=upper_cycle_limit,
        lambda_nonneg=lambda_nonneg,
        lambda_upper=lambda_upper,
        lambda_fs=lambda_fs,
        lambda_mech=lambda_mech,
        fp_train=fp_all[train_idx],
        mech_labels_train=mech_all[train_idx],
        material_name=material,
        device=device,
        grad_clip_norm=1.0,
    )

    # 预测（输出 cycles 与机制概率）
    Np_tr, Mech_tr = predict_outputs(model, Xz[train_idx], device=device)
    Np_te, Mech_te = predict_outputs(model, Xz[test_idx], device=device)

    evaluate_and_save_split(material, dataset, train_idx, test_idx, Np_tr, Np_te, Mech_tr, Mech_te)


def run_material_loo(
    material: str,
    *,
    include_fp: bool,
    hidden_dims: list[int],
    dropout: float,
    lr: float,
    weight_decay: float,
    epochs: int,
    batch_size: int,
    upper_cycle_limit: float,
    lambda_nonneg: float,
    lambda_upper: float,
    lambda_fs: float,
    lambda_mech: float,
    seed: int,
    device: torch.device,
):
    print(f'开始处理材料: {material} (MFLP-PINN + 时间序列特征，LOO)')
    dataset = build_dataset(material, include_fp=include_fp)
    X = dataset['X']
    y_log = dataset['y']
    mu = dataset['mu']
    sigma = dataset['sigma']

    sigma_safe = np.where(sigma > 0, sigma, 1.0)
    Xz = (X - mu) / sigma_safe

    n = Xz.shape[0]
    fp_all = np.array([m['FP'] for m in dataset['meta']], dtype=np.float64)
    mech_all = np.array([
        generate_soft_mechanism_label(material, float(m['gamma_a']), float(m['epsilon_a']))
        for m in dataset['meta']
    ], dtype=np.float64)
    Np_pred_all = np.zeros(n, dtype=np.float64)
    Mech_prob_all = np.zeros(n, dtype=np.float64)

    rng = np.random.RandomState(int(seed))
    order = np.arange(n)
    rng.shuffle(order)

    # 为了效率，LOO 每次重新初始化并短训练
    for count, i in enumerate(order):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        model = train_mflp_pinn(
            X_train=Xz[mask],
            y_train_log10=y_log[mask],
            hidden_dims=hidden_dims,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay,
            epochs=epochs,
            batch_size=batch_size,
            upper_cycle_limit=upper_cycle_limit,
            lambda_nonneg=lambda_nonneg,
            lambda_upper=lambda_upper,
            lambda_fs=lambda_fs,
            lambda_mech=lambda_mech,
            fp_train=fp_all[mask],
            mech_labels_train=mech_all[mask],
            material_name=material,
            device=device,
            grad_clip_norm=1.0,
        )
        Np_pred_i, Mech_prob_i = predict_outputs(model, Xz[i:i+1], device=device)
        Np_pred_all[i] = float(Np_pred_i[0])
        Mech_prob_all[i] = float(Mech_prob_i[0])
        if (count + 1) % max(1, n // 10) == 0:
            print(f'LOO 进度: {count + 1}/{n}')

    evaluate_and_save_loo(material, dataset, Np_pred_all, Mech_prob_all)


def parse_hidden_dims(text: str) -> list[int]:
    if text.strip() == '':
        return []
    return [int(x) for x in text.split(',') if x.strip()]


def main():
    import argparse
    parser = argparse.ArgumentParser(description='MFLP-PINN：基于物理约束的多轴疲劳寿命预测（时间序列特征）')
    parser.add_argument('--material', type=str, default='ALL', choices=['ALL'] + list(DATA_DIRS.keys()))
    parser.add_argument('--no-fp', action='store_true', help='不使用疲劳参数 FP 作为特征')
    parser.add_argument('--method', type=str, default='split', choices=['split', 'loo'], help='预测方式：split 或 loo')
    parser.add_argument('--test-ratio', type=float, default=0.2, help='测试集比例（split 模式）')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')

    # 模型/训练参数
    parser.add_argument('--hidden-dims', type=str, default='128,64', help='隐藏层维度，例如 128,64')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout 比例')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--epochs', type=int, default=1500, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=32, help='批大小')
    parser.add_argument('--upper-cycles', type=float, default=1e7, help='疲劳寿命上限（物理约束）')
    parser.add_argument('--lambda-nonneg', type=float, default=1.0, help='非负约束权重')
    parser.add_argument('--lambda-upper', type=float, default=0.1, help='上界约束权重')
    parser.add_argument('--lambda-fs', type=float, default=0.1, help='FS 物理损失权重')
    parser.add_argument('--lambda-mech', type=float, default=0.3, help='机制分类软标签损失权重')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'])

    args = parser.parse_args()

    seed_everything(int(args.seed))
    dev = torch.device('cuda' if (args.device == 'cuda' or (args.device == 'auto' and torch.cuda.is_available())) else 'cpu')
    hidden_dims = parse_hidden_dims(args.hidden_dims)

    include_fp = (not args.no_fp)

    if args.material == 'ALL':
        materials = get_available_materials()
        if not materials:
            print('未找到任何材料的应变时间序列数据，无法运行。')
            sys.exit(1)
        print(f"将处理以下材料: {', '.join(materials)}")
        for mat in materials:
            try:
                if args.method == 'loo':
                    run_material_loo(
                        mat,
                        include_fp=include_fp,
                        hidden_dims=hidden_dims,
                        dropout=float(args.dropout),
                        lr=float(args.lr),
                        weight_decay=float(args.weight_decay),
                        epochs=int(max(1, args.epochs // 3)),  # LOO 适当减小训练轮数
                        batch_size=int(args.batch_size),
                        upper_cycle_limit=float(args.upper_cycles),
                        lambda_nonneg=float(args.lambda_nonneg),
                        lambda_upper=float(args.lambda_upper),
                        lambda_fs=float(args.lambda_fs),
                        lambda_mech=float(args.lambda_mech),
                        seed=int(args.seed),
                        device=dev,
                    )
                else:
                    run_material_split(
                        mat,
                        include_fp=include_fp,
                        hidden_dims=hidden_dims,
                        dropout=float(args.dropout),
                        lr=float(args.lr),
                        weight_decay=float(args.weight_decay),
                        epochs=int(args.epochs),
                        batch_size=int(args.batch_size),
                        upper_cycle_limit=float(args.upper_cycles),
                        lambda_nonneg=float(args.lambda_nonneg),
                        lambda_upper=float(args.lambda_upper),
                        lambda_fs=float(args.lambda_fs),
                        lambda_mech=float(args.lambda_mech),
                        test_ratio=float(args.test_ratio),
                        seed=int(args.seed),
                        device=dev,
                    )
            except Exception as e:
                print(f"处理 {mat} 时发生错误: {e}")
    else:
        if args.method == 'loo':
            run_material_loo(
                args.material,
                include_fp=include_fp,
                hidden_dims=hidden_dims,
                dropout=float(args.dropout),
                lr=float(args.lr),
                weight_decay=float(args.weight_decay),
                epochs=int(max(1, args.epochs // 3)),
                batch_size=int(args.batch_size),
                upper_cycle_limit=float(args.upper_cycles),
                lambda_nonneg=float(args.lambda_nonneg),
                lambda_upper=float(args.lambda_upper),
                lambda_fs=float(args.lambda_fs),
                lambda_mech=float(args.lambda_mech),
                seed=int(args.seed),
                device=dev,
            )
        else:
            run_material_split(
                args.material,
                include_fp=include_fp,
                hidden_dims=hidden_dims,
                dropout=float(args.dropout),
                lr=float(args.lr),
                weight_decay=float(args.weight_decay),
                epochs=int(args.epochs),
                batch_size=int(args.batch_size),
                upper_cycle_limit=float(args.upper_cycles),
                lambda_nonneg=float(args.lambda_nonneg),
                lambda_upper=float(args.lambda_upper),
                lambda_fs=float(args.lambda_fs),
                lambda_mech=float(args.lambda_mech),
                test_ratio=float(args.test_ratio),
                seed=int(args.seed),
                device=dev,
            )


if __name__ == '__main__':
    main()


