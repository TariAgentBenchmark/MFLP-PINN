import os

import argparse
import sys
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F


# 切到脚本所在目录，确保相对路径正确
script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
os.chdir(script_dir)


# 数据目录（与 main.py 保持一致）
DATA_DIRS = {
    'AISI316L': {
        'strain_series': 'data/AISI316L应变时间序列数据',
        'fatigue_data': 'data/多轴疲劳试验数据/AISI316L多轴疲劳试验数据.xls'
    },
    'GH4169': {
        'strain_series': 'data/GH4169应变时间序列数据',
        'fatigue_data': 'data/多轴疲劳试验数据/GH4169多轴疲劳试验数据.xls'
    },
    'TC4': {
        'strain_series': 'data/TC4应变时间序列数据',
        'fatigue_data': 'data/多轴疲劳试验数据/TC4多轴疲劳试验数据.xls'
    },
    'CuZn37': {
        'strain_series': 'data/CuZn37应变时间序列数据',
        'fatigue_data': 'data/多轴疲劳试验数据/CuZn37黄铜多轴疲劳试验数据.xls'
    },
    'Q235B1': {
        'strain_series': 'data/Q235B基础金属应变时间序列数据',
        'fatigue_data': 'data/多轴疲劳试验数据/Q235B1.xls'
    },
    'Q235B2': {
        'strain_series': 'data/Q235B焊接金属应变时间序列数据',
        'fatigue_data': 'data/多轴疲劳试验数据/Q235B2.xls'
    }
}


def get_strain_series_path(material_name: str) -> str:
    return DATA_DIRS[material_name]['strain_series']


def get_fatigue_data_path(material_name: str) -> str:
    return DATA_DIRS[material_name]['fatigue_data']


def get_output_path(material_name: str, file_name: str) -> str:
    base_dir = os.path.join(os.getcwd(), 'results')
    os.makedirs(base_dir, exist_ok=True)
    material_dir = os.path.join(base_dir, material_name)
    os.makedirs(material_dir, exist_ok=True)
    return os.path.join(material_dir, file_name)


def parse_strain_values_from_filename(filename: str):
    parts = filename.replace('strain_series_', '').replace('.xls', '').replace('.csv', '').split('_')
    return float(parts[0]), float(parts[1])


def read_strain_series(file_path: str):
    if os.path.getsize(file_path) == 0:
        raise ValueError('文件为空')
    if file_path.endswith('.csv'):
        encodings = ['utf-8', 'gbk', 'gb2312', 'latin1']
        df = None
        for enc in encodings:
            try:
                df = pd.read_csv(file_path, encoding=enc)
                break
            except UnicodeDecodeError:
                pass
        if df is None:
            raise ValueError('CSV 编码无法识别')
    else:
        df = pd.read_excel(file_path)

    required_columns = ['Time (s)', 'Normal Strain', 'Shear Strain']
    for c in required_columns:
        if c not in df.columns:
            raise ValueError(f'缺少必要列: {c}')

    time = df['Time (s)'].astype(np.float64).to_numpy()
    normal = df['Normal Strain'].astype(np.float64).to_numpy()
    shear = df['Shear Strain'].astype(np.float64).to_numpy()

    if np.any(~np.isfinite(time)) or np.any(~np.isfinite(normal)) or np.any(~np.isfinite(shear)):
        raise ValueError('数据包含非有限值')

    return time, normal, shear


def read_fatigue_data(file_path: str):
    excel = pd.ExcelFile(file_path)
    epsilon_a, gamma_a, Nf, FP = [], [], [], []
    for sheet in excel.sheet_names:
        df = pd.read_excel(file_path, sheet_name=sheet)
        if df.shape[1] < 6:
            continue
        epsilon_a.extend(df.iloc[:, 0].to_numpy())
        gamma_a.extend(df.iloc[:, 1].to_numpy())
        Nf.extend(df.iloc[:, 2].to_numpy())
        FP.extend(df.iloc[:, 5].to_numpy())
    return np.array(epsilon_a, dtype=np.float64), np.array(gamma_a, dtype=np.float64), np.array(Nf, dtype=np.float64), np.array(FP, dtype=np.float64)


# 特征工程
def _safe_stat(x: np.ndarray, fn):
    v = fn(x)
    return float(v) if np.isfinite(v) else 0.0


def _skewness(x: np.ndarray) -> float:
    n = x.size
    if n < 3:
        return 0.0
    xm = x - np.mean(x)
    s = np.std(xm)
    if s == 0.0:
        return 0.0
    m3 = np.mean(xm ** 3)
    return float(m3 / (s ** 3))


def _kurtosis(x: np.ndarray) -> float:
    n = x.size
    if n < 4:
        return 0.0
    xm = x - np.mean(x)
    s = np.std(xm)
    if s == 0.0:
        return 0.0
    m4 = np.mean(xm ** 4)
    return float(m4 / (s ** 4) - 3.0)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2)))


def _autocorr_lag1(x: np.ndarray) -> float:
    if x.size < 2:
        return 0.0
    x0 = x[:-1]
    x1 = x[1:]
    s0 = np.std(x0)
    s1 = np.std(x1)
    if s0 == 0 or s1 == 0:
        return 0.0
    return float(np.corrcoef(x0, x1)[0, 1])


def _corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    sx = np.std(x)
    sy = np.std(y)
    if sx == 0 or sy == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _fft_features(time: np.ndarray, x: np.ndarray):
    n = x.size
    if n < 8:
        return 0.0, 0.0, 0.0
    dt_raw = np.diff(time)
    dt = float(np.median(dt_raw)) if dt_raw.size > 0 else 0.0
    if not np.isfinite(dt) or dt <= 0:
        return 0.0, 0.0, 0.0
    x_d = x - np.mean(x)
    X = np.fft.rfft(x_d)
    freqs = np.fft.rfftfreq(n, d=dt)
    P = np.abs(X) ** 2
    if P.sum() <= 0.0:
        return 0.0, 0.0, 0.0
    # 主频率（去掉直流分量）
    if P.size > 1:
        idx = 1 + int(np.argmax(P[1:]))
        f_dom = float(freqs[idx])
    else:
        f_dom = 0.0
    # 频谱质心
    centroid = float(np.sum(freqs * P) / np.sum(P))
    # 频谱熵（归一化到 [0,1]）
    p = P / np.sum(P)
    p = np.where(p > 0, p, 1e-12)
    entropy = float(-np.sum(p * np.log(p)) / np.log(p.size))
    return f_dom, centroid, entropy


def compute_features(time: np.ndarray, normal: np.ndarray, shear: np.ndarray) -> np.ndarray:
    # 基础统计
    feats = []
    for arr in (normal, shear):
        feats.append(_safe_stat(arr, np.mean))
        feats.append(_safe_stat(arr, np.std))
        feats.append(_safe_stat(arr, np.min))
        feats.append(_safe_stat(arr, np.max))
        feats.append(float(np.max(arr) - np.min(arr)))
        feats.append(_rms(arr))
        feats.append(_skewness(arr))
        feats.append(_kurtosis(arr))
        feats.append(_autocorr_lag1(arr))
        f_dom, f_cent, f_ent = _fft_features(time, arr)
        feats.extend([f_dom, f_cent, f_ent])
    # 交互特征
    feats.append(_corrcoef(normal, shear))
    return np.array(feats, dtype=np.float64)


# 构建数据集（按材料）
def build_dataset(material_name: str, include_fp: bool = True):
    strain_dir = get_strain_series_path(material_name)
    fatigue_path = get_fatigue_data_path(material_name)
    if not os.path.exists(strain_dir):
        raise FileNotFoundError(f'找不到应变时间序列文件夹: {strain_dir}')
    if not os.path.exists(fatigue_path):
        raise FileNotFoundError(f'找不到疲劳数据文件: {fatigue_path}')

    # 预扫描时间序列 -> 特征
    feature_map = {}
    file_map = {}
    for fname in os.listdir(strain_dir):
        if not (fname.startswith('strain_series_') and (fname.endswith('.xls') or fname.endswith('.csv'))):
            continue
        ea, ga = parse_strain_values_from_filename(fname)
        key = (round(float(ea), 5), round(float(ga), 5))
        path = os.path.join(strain_dir, fname)
        time, normal, shear = read_strain_series(path)
        feats = compute_features(time, normal, shear)
        feature_map[key] = feats
        file_map[key] = fname

    eps_all, gam_all, Nf_all, FP_all = read_fatigue_data(fatigue_path)

    X_list, y_list = [], []
    meta = []
    for i in range(eps_all.size):
        key = (round(float(eps_all[i]), 5), round(float(gam_all[i]), 5))
        if key not in feature_map:
            continue
        feats = feature_map[key]
        # 将幅值与 FP 作为额外特征
        extra = [float(eps_all[i]), float(gam_all[i])]
        if include_fp:
            extra.append(float(FP_all[i]))
        x = np.concatenate([feats, np.array(extra, dtype=np.float64)])
        X_list.append(x)
        y_list.append(math.log10(float(Nf_all[i])))
        meta.append({
            'file': file_map.get(key, ''),
            'epsilon_a': float(eps_all[i]),
            'gamma_a': float(gam_all[i]),
            'FP': float(FP_all[i]),
            'Nf': float(Nf_all[i])
        })

    if len(X_list) == 0:
        raise ValueError('没有任何匹配的数据对（时间序列 与 疲劳表格）')

    X = np.vstack(X_list)
    y = np.array(y_list, dtype=np.float64)

    # 标准化（全局）
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma = np.where(sigma > 0, sigma, 1.0)
    Xz = (X - mu) / sigma
    # 设计矩阵（加截距）
    Phi = np.concatenate([np.ones((Xz.shape[0], 1)), Xz], axis=1)

    return {
        'Phi': Phi,
        'X': X,
        'y': y,
        'mu': mu,
        'sigma': sigma,
        'meta': meta
    }


def train_test_split_indices(n: int, test_ratio: float = 0.2, seed: int = 42):
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    t = int(round(n * (1.0 - test_ratio)))
    train_idx = np.sort(idx[:t])
    test_idx = np.sort(idx[t:])
    return train_idx, test_idx


def get_available_materials():
    mats = []
    for material_name in DATA_DIRS.keys():
        strain_series_path = get_strain_series_path(material_name)
        if os.path.exists(strain_series_path):
            files = [f for f in os.listdir(strain_series_path)
                     if f.startswith('strain_series_') and (f.endswith('.xls') or f.endswith('.csv'))]
            if files:
                mats.append(material_name)
    return mats


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


def _choose_transformer_heads(d_model: int) -> int:
    for h in [8, 4, 2, 1]:
        if d_model % h == 0:
            return h
    return 1


def _resolve_transformer_dims(hidden_dims: list[int]) -> tuple[int, int, int]:
    # d_model kept small for parameter efficiency
    d_model = min(max(16, (hidden_dims[0] if hidden_dims else 64)), 64)
    num_layers = max(1, min(len(hidden_dims), 2))
    dim_feedforward = hidden_dims[1] if len(hidden_dims) >= 2 else max(64, 4 * d_model)
    return d_model, num_layers, dim_feedforward


def _build_sinusoidal_positional_encoding(seq_len: int, d_model: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    position = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device, dtype=dtype) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(seq_len, d_model, device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class TransformerBackbone(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float):
        super().__init__()
        d_model, num_layers, dim_feedforward = _resolve_transformer_dims(hidden_dims)
        nhead = _choose_transformer_heads(d_model)
        self.input_dim = int(input_dim)
        self.d_model = int(d_model)
        self.input_proj = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=int(dim_feedforward),
            dropout=float(dropout),
            activation='relu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(num_layers))
        self.dropout = nn.Dropout(p=float(dropout)) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, input_dim)
        batch_size, feat_dim = x.shape
        if feat_dim != self.input_dim:
            # Guard against mismatch; truncate or pad with zeros to expected length (keeps interface robust)
            if feat_dim > self.input_dim:
                x = x[:, : self.input_dim]
            else:
                pad = torch.zeros(batch_size, self.input_dim - feat_dim, device=x.device, dtype=x.dtype)
                x = torch.cat([x, pad], dim=1)
        seq = x.unsqueeze(-1)  # (B, L, 1)
        token = self.input_proj(seq)  # (B, L, d_model)
        pe = _build_sinusoidal_positional_encoding(self.input_dim, self.d_model, token.device, token.dtype)
        token = token + pe.unsqueeze(0)
        h = self.encoder(token)  # (B, L, d_model)
        h = self.dropout(h.mean(dim=1))  # mean-pool over feature positions -> (B, d_model)
        return h


class SmallTransformerRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float):
        super().__init__()
        self.backbone = TransformerBackbone(input_dim, hidden_dims, dropout)
        self.head = nn.Linear(self.backbone.d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        return self.head(h)


class MLPBackbone(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=float(dropout)))
            prev_dim = hidden_dim
        self.network = nn.Sequential(*layers) if layers else nn.Identity()
        self.output_dim = prev_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.network(x)
        return h


class SmallMLPRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float):
        super().__init__()
        self.backbone = MLPBackbone(input_dim, hidden_dims, dropout)
        self.head = nn.Linear(self.backbone.output_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        return self.head(h)


def build_mlp(input_dim: int, hidden_dims: list[int], dropout: float) -> nn.Sequential:
    # Return an actual MLP regressor for backward compatibility with the name
    return nn.Sequential(SmallMLPRegressor(input_dim, hidden_dims, dropout))


class MultiTaskMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float, backbone: str = 'transformer'):
        super().__init__()
        backbone = (backbone or 'transformer').lower()
        if backbone == 'mlp':
            bb = MLPBackbone(input_dim, hidden_dims, dropout)
            out_dim = bb.output_dim
        else:
            bb = TransformerBackbone(input_dim, hidden_dims, dropout)
            out_dim = bb.d_model
        self.backbone = bb
        self.life_head = nn.Linear(out_dim, 1)
        self.mech_head = nn.Linear(out_dim, 1)

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
    mtl_enabled: bool = True,
    backbone: str = 'transformer',
    # 可选验证集，用于绘制 loss 曲线
    X_val: np.ndarray | None = None,
    y_val_log10: np.ndarray | None = None,
    fp_val: np.ndarray | None = None,
    mech_labels_val: np.ndarray | None = None,
):
    model: nn.Module
    if mtl_enabled:
        model = MultiTaskMLP(X_train.shape[1], hidden_dims, dropout, backbone=backbone).to(device)
    else:
        if (backbone or 'transformer').lower() == 'mlp':
            model = build_mlp(X_train.shape[1], hidden_dims, dropout).to(device)
        else:
            model = SmallTransformerRegressor(X_train.shape[1], hidden_dims, dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    X_tr = torch.from_numpy(X_train.astype(np.float32)).to(device)
    y_tr_log = torch.from_numpy(y_train_log10.astype(np.float32)).to(device)
    FP_tr = torch.from_numpy(fp_train.astype(np.float32)).to(device)
    mech_tr = torch.from_numpy(mech_labels_train.astype(np.float32)).to(device)

    num_samples = X_tr.shape[0]
    num_batches = max(1, int(math.ceil(num_samples / max(1, batch_size))))

    # 记录损失历史
    history = {
        'train_total': [],
        'val_total': [],
        'train_data': [],
        'val_data': [],
        'train_phys': [],
        'val_phys': [],
        'train_mech': [],
        'val_mech': [],
    }

    model.train()
    for epoch in range(int(epochs)):
        perm = torch.randperm(num_samples, device=device)
        epoch_loss = 0.0
        epoch_data = 0.0
        epoch_phys = 0.0
        epoch_mech = 0.0
        for b in range(num_batches):
            idx = perm[b * batch_size : (b + 1) * batch_size]
            x_b = X_tr.index_select(0, idx)
            y_b_log = y_tr_log.index_select(0, idx)

            if mtl_enabled:
                # 模型输出：预测 log10(Np) 与机制概率
                mu_log_pred, mech_prob = model(x_b)
            else:
                # 单任务：仅预测 log10(Np)
                mu_log_pred = model(x_b).squeeze(-1)

            # 将 log10(Np) 指标还原为线性 Np 以用于物理项
            Np_pred = torch.pow(10.0, mu_log_pred)

            # 主损失：直接在 log10 空间与真实值进行 MSE（数值更稳定）
            loss_mse = F.mse_loss(mu_log_pred, y_b_log)

            # 物理约束：非负与上界（对线性 Np 约束，例如 1e7 次循环）
            loss_phys_low = F.relu(-Np_pred).mean() * float(lambda_nonneg)
            loss_phys_high = F.relu(Np_pred - float(upper_cycle_limit)).mean() * float(lambda_upper)

            # FS 物理损失（基于线性 Np）
            FP_b = FP_tr.index_select(0, idx)
            loss_fs_val = fs_loss(Np_pred, FP_b, material_name) * float(lambda_fs)

            if mtl_enabled:
                # 机制分类软标签损失（仅对有效标签计算）
                mech_b = mech_tr.index_select(0, idx)
                valid_mask = mech_b >= 0.0
                if torch.any(valid_mask):
                    loss_mech = soft_binary_cross_entropy(mech_prob[valid_mask], mech_b[valid_mask]) * float(lambda_mech)
                else:
                    loss_mech = torch.tensor(0.0, device=device)
                loss = loss_mse + loss_phys_low + loss_phys_high + loss_fs_val + loss_mech
            else:
                loss = loss_mse + loss_phys_low + loss_phys_high + loss_fs_val

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip_norm is not None and grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
            optimizer.step()

            epoch_loss += float(loss.detach().cpu())
            epoch_data += float(loss_mse.detach().cpu())
            epoch_phys += float((loss_phys_low + loss_phys_high + loss_fs_val).detach().cpu())
            epoch_mech += float((loss_mech if mtl_enabled else torch.tensor(0.0, device=device)).detach().cpu())

        # 可选打印
        if (epoch + 1) % max(1, int(epochs // 10)) == 0:
            print(f'Epoch {epoch + 1}/{epochs} - loss: {epoch_loss / num_batches:.6f}')

        # 记录训练集 epoch 平均
        history['train_total'].append(epoch_loss / num_batches)
        history['train_data'].append(epoch_data / num_batches)
        history['train_phys'].append(epoch_phys / num_batches)
        history['train_mech'].append(epoch_mech / num_batches)

        # 验证损失（若提供）
        if X_val is not None and y_val_log10 is not None and fp_val is not None:
            with torch.no_grad():
                model.eval()
                Xv = torch.from_numpy(X_val.astype(np.float32)).to(device)
                yv = torch.from_numpy(y_val_log10.astype(np.float32)).to(device)
                FPv = torch.from_numpy(fp_val.astype(np.float32)).to(device)
                if mtl_enabled and mech_labels_val is not None:
                    Mechv = torch.from_numpy(mech_labels_val.astype(np.float32)).to(device)
                else:
                    Mechv = None

                if isinstance(model, MultiTaskMLP):
                    mu_log_v, mech_prob_v = model(Xv)
                else:
                    mu_log_v = model(Xv).squeeze(-1)
                    mech_prob_v = torch.zeros_like(mu_log_v)
                Np_v = torch.pow(10.0, mu_log_v)
                v_data = F.mse_loss(mu_log_v, yv)
                v_low = F.relu(-Np_v).mean() * float(lambda_nonneg)
                v_high = F.relu(Np_v - float(upper_cycle_limit)).mean() * float(lambda_upper)
                v_fs = fs_loss(Np_v, FPv, material_name) * float(lambda_fs)
                if mtl_enabled and Mechv is not None:
                    v_mask = Mechv >= 0.0
                    if torch.any(v_mask):
                        v_mech = soft_binary_cross_entropy(mech_prob_v[v_mask], Mechv[v_mask]) * float(lambda_mech)
                    else:
                        v_mech = torch.tensor(0.0, device=device)
                else:
                    v_mech = torch.tensor(0.0, device=device)
                v_phys = v_low + v_high + v_fs
                v_total = v_data + v_phys + v_mech
                history['val_total'].append(float(v_total.detach().cpu()))
                history['val_data'].append(float(v_data.detach().cpu()))
                history['val_phys'].append(float(v_phys.detach().cpu()))
                history['val_mech'].append(float(v_mech.detach().cpu()))
                model.train()

    model.eval()
    return model, history


def predict_cycles(model: nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        X_t = torch.from_numpy(X.astype(np.float32)).to(device)
        # 模型输出的是 mu_log = log10(Np)
        if isinstance(model, MultiTaskMLP):
            mu_log, _ = model(X_t)
        else:
            mu_log = model(X_t).squeeze(-1)
        Np_pred = torch.pow(10.0, mu_log)
        return Np_pred.detach().cpu().numpy().astype(np.float64)


def predict_outputs(model: nn.Module, X: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        X_t = torch.from_numpy(X.astype(np.float32)).to(device)
        # 输出 mu_log = log10(Np)，再转回线性空间
        if isinstance(model, MultiTaskMLP):
            mu_log, mech_prob = model(X_t)
            Np_pred = torch.pow(10.0, mu_log)
        else:
            mu_log = model(X_t).squeeze(-1)
            Np_pred = torch.pow(10.0, mu_log)
            mech_prob = torch.zeros_like(mu_log)
        return (
            Np_pred.detach().cpu().numpy().astype(np.float64),
            mech_prob.detach().cpu().numpy().astype(np.float64),
        )


def plot_loss_history(material: str, history: dict):
    # 若没有有效历史，直接返回
    if not history or len(history.get('train_total', [])) == 0:
        return
    epochs = np.arange(1, len(history['train_total']) + 1)
    plt.figure(figsize=(10, 6))
    # 训练与验证总损失
    plt.plot(epochs, history['train_total'], color='blue', linewidth=2.0, label='Train Total Loss')
    if len(history.get('val_total', [])) == len(epochs):
        plt.plot(epochs, history['val_total'], color='red', linewidth=2.0, label='Val Total Loss')
    # 数据项
    if len(history.get('train_data', [])) == len(epochs):
        plt.plot(epochs, history['train_data'], color='blue', linestyle='--', linewidth=1.5, label='Train Data Loss')
    if len(history.get('val_data', [])) == len(epochs):
        plt.plot(epochs, history['val_data'], color='red', linestyle='--', linewidth=1.5, label='Val Data Loss')
    # 物理项
    if len(history.get('train_phys', [])) == len(epochs):
        plt.plot(epochs, history['train_phys'], color='green', linestyle=':', linewidth=2.0, label='Train Physical Loss')
    if len(history.get('val_phys', [])) == len(epochs):
        plt.plot(epochs, history['val_phys'], color='green', linestyle=':', linewidth=2.0, alpha=0.8, label='Val Physical Loss')
    # 机制项
    if len(history.get('train_mech', [])) == len(epochs):
        plt.plot(epochs, history['train_mech'], color='cyan', linestyle='--', linewidth=1.5, label='Train Mech Loss')
    if len(history.get('val_mech', [])) == len(epochs):
        plt.plot(epochs, history['val_mech'], color='magenta', linestyle='--', linewidth=1.5, label='Val Mech Loss')

    plt.xlabel('Epochs')
    plt.ylabel('Loss Value')
    plt.title(f'{material} Training Loss History')
    plt.grid(True, alpha=0.5)
    plt.legend()
    plot_path = get_output_path(material, f'{material}_loss_history.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'训练/验证损失曲线已保存: {plot_path}')


def _compute_binary_metrics(y_true_soft: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    """针对机制预测的软标签计算主要二分类指标。"""
    mask = y_true_soft >= 0.0
    valid_scores = y_score[mask]
    valid_labels = y_true_soft[mask]
    support = int(valid_labels.size)
    if support == 0:
        return {'auc': float('nan'), 'accuracy': float('nan'), 'support': 0}

    y_true = (valid_labels >= 0.5).astype(np.int32)
    y_pred = (valid_scores >= 0.5).astype(np.int32)
    accuracy = float(np.mean(y_pred == y_true)) if support > 0 else float('nan')

    pos = int(y_true.sum())
    neg = support - pos
    if pos == 0 or neg == 0:
        auc = float('nan')
    else:
        # 以秩统计计算 ROC AUC，避免依赖外部库
        order = np.argsort(valid_scores)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, support + 1)
        sum_ranks_pos = float(np.sum(ranks[y_true == 1]))
        auc = (sum_ranks_pos - pos * (pos + 1) / 2.0) / (pos * neg)

    return {'auc': auc, 'accuracy': accuracy, 'support': support}


def plot_mechanism_metrics(material: str, metrics_by_split: dict[str, dict[str, float]], mode: str):
    """绘制机制预测的 AUC 与准确率条形图。"""
    if not metrics_by_split:
        return

    splits = list(metrics_by_split.keys())
    metrics_names = ['auc', 'accuracy']
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e']

    fig, axes = plt.subplots(1, len(metrics_names), figsize=(12, 5))
    axes = np.atleast_1d(axes)
    for ax, metric in zip(axes, metrics_names):
        values = [metrics_by_split[split].get(metric, float('nan')) for split in splits]
        finite_vals = [v for v in values if np.isfinite(v)]
        positions = np.arange(len(splits), dtype=np.float64)
        bars = ax.bar(
            positions,
            [v if np.isfinite(v) else 0.0 for v in values],
            color=[colors[i % len(colors)] for i in range(len(splits))],
            alpha=0.85,
        )
        for bar, v in zip(bars, values):
            x = bar.get_x() + bar.get_width() / 2.0
            if np.isfinite(v):
                ax.text(x, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=10)
            else:
                ax.text(x, 0.02, 'N/A', ha='center', va='bottom', fontsize=10)
        ax.set_ylim(0.0, min(1.05, max(1.0, max(finite_vals, default=1.0) + 0.05)))
        ax.set_ylabel(metric.upper())
        ax.set_xticks(positions)
        ax.set_xticklabels(splits, rotation=20)
        ax.set_title(f'{metric.upper()}')
        ax.grid(True, axis='y', alpha=0.3)

    fig.suptitle(f'{material} Mechanism Prediction Metrics ({mode})')
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    plot_path = get_output_path(material, f'{material}_mechanism_metrics_{mode}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'机制预测指标图已保存: {plot_path}')


def evaluate_and_save_split(
    material: str,
    dataset: dict,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    Np_tr: np.ndarray,
    Np_te: np.ndarray,
    Mech_tr: np.ndarray,
    Mech_te: np.ndarray,
    mech_labels_train: np.ndarray,
    mech_labels_test: np.ndarray,
    mtl_enabled: bool,
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
    plt.plot([mn, mx], [mn, mx], color='#d62728', linestyle='--', linewidth=1.8, label='Perfect')
    # Factor of 1.5 (green dashdot)
    plt.plot([mn, mx], [mn / 1.5, mx / 1.5], color='#2ca02c', linestyle='-.', linewidth=2.0, label='Factor of 1.5')
    plt.plot([mn, mx], [mn * 1.5, mx * 1.5], color='#2ca02c', linestyle='-.', linewidth=2.0)
    # Factor of 2 (black dotted, thicker)
    plt.plot([mn, mx], [mn / 2.0, mx / 2.0], color='#000000', linestyle=':', linewidth=2.4, label='Factor of 2')
    plt.plot([mn, mx], [mn * 2.0, mx * 2.0], color='#000000', linestyle=':', linewidth=2.4)
    # Factor of 3 (purple solid)
    plt.plot([mn, mx], [mn / 3.0, mx / 3.0], color='#9467bd', linestyle='-', linewidth=2.0, label='Factor of 3')
    plt.plot([mn, mx], [mn * 3.0, mx * 3.0], color='#9467bd', linestyle='-', linewidth=2.0)
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

    if mtl_enabled:
        train_metrics = _compute_binary_metrics(np.asarray(mech_labels_train, dtype=np.float64), np.asarray(Mech_tr, dtype=np.float64))
        test_metrics = _compute_binary_metrics(np.asarray(mech_labels_test, dtype=np.float64), np.asarray(Mech_te, dtype=np.float64))

        train_auc = f"{train_metrics['auc']:.3f}" if np.isfinite(train_metrics['auc']) else 'NA'
        train_acc = f"{train_metrics['accuracy']:.3f}" if np.isfinite(train_metrics['accuracy']) else 'NA'
        test_auc = f"{test_metrics['auc']:.3f}" if np.isfinite(test_metrics['auc']) else 'NA'
        test_acc = f"{test_metrics['accuracy']:.3f}" if np.isfinite(test_metrics['accuracy']) else 'NA'

        print(f"机制预测-训练集: 样本={train_metrics['support']}, AUC={train_auc}, 准确率={train_acc}")
        print(f"机制预测-测试集: 样本={test_metrics['support']}, AUC={test_auc}, 准确率={test_acc}")

        if (train_metrics['support'] > 0) or (test_metrics['support'] > 0):
            plot_mechanism_metrics(
                material,
                {
                    'Train': train_metrics,
                    'Test': test_metrics,
                },
                mode='split',
            )


def evaluate_and_save_loo(
    material: str,
    dataset: dict,
    Np_pred_all: np.ndarray,
    Mech_prob_all: np.ndarray,
    mech_labels_all: np.ndarray,
    mtl_enabled: bool,
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
    plt.plot([mn, mx], [mn, mx], color='#d62728', linestyle='--', linewidth=1.8, label='Perfect')
    # Factor of 1.5 (green dashdot)
    plt.plot([mn, mx], [mn / 1.5, mx / 1.5], color='#2ca02c', linestyle='-.', linewidth=2.0, label='Factor of 1.5')
    plt.plot([mn, mx], [mn * 1.5, mx * 1.5], color='#2ca02c', linestyle='-.', linewidth=2.0)
    # Factor of 2 (black dotted, thicker)
    plt.plot([mn, mx], [mn / 2.0, mx / 2.0], color='#000000', linestyle=':', linewidth=2.4, label='Factor of 2')
    plt.plot([mn, mx], [mn * 2.0, mx * 2.0], color='#000000', linestyle=':', linewidth=2.4)
    # Factor of 3 (purple solid)
    plt.plot([mn, mx], [mn / 3.0, mx / 3.0], color='#9467bd', linestyle='-', linewidth=2.0, label='Factor of 3')
    plt.plot([mn, mx], [mn * 3.0, mx * 3.0], color='#9467bd', linestyle='-', linewidth=2.0)
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

    if mtl_enabled:
        all_metrics = _compute_binary_metrics(np.asarray(mech_labels_all, dtype=np.float64), np.asarray(Mech_prob_all, dtype=np.float64))
        if all_metrics['support'] > 0:
            auc_info = f"{all_metrics['auc']:.3f}" if np.isfinite(all_metrics['auc']) else 'NA'
            acc_info = f"{all_metrics['accuracy']:.3f}" if np.isfinite(all_metrics['accuracy']) else 'NA'
            print(f"机制预测-整体: 样本={all_metrics['support']}, AUC={auc_info}, 准确率={acc_info}")
            plot_mechanism_metrics(
                material,
                {
                    'All': all_metrics,
                },
                mode='loo',
            )
        else:
            print('机制预测-整体: 无可用软标签，跳过 AUC/准确率评估')


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
    mtl_enabled: bool,
    backbone: str,
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
    model, history = train_mflp_pinn(
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
        mtl_enabled=mtl_enabled,
        backbone=backbone,
        # 验证集（用于绘制曲线）
        X_val=Xz[test_idx],
        y_val_log10=y_log[test_idx],
        fp_val=fp_all[test_idx],
        mech_labels_val=mech_all[test_idx] if mtl_enabled else None,
    )

    # 预测（输出 cycles 与机制概率）
    Np_tr, Mech_tr = predict_outputs(model, Xz[train_idx], device=device)
    Np_te, Mech_te = predict_outputs(model, Xz[test_idx], device=device)

    evaluate_and_save_split(
        material,
        dataset,
        train_idx,
        test_idx,
        Np_tr,
        Np_te,
        Mech_tr,
        Mech_te,
        mech_all[train_idx],
        mech_all[test_idx],
        mtl_enabled,
    )
    # 绘制并保存 loss 曲线
    plot_loss_history(material, history)


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
    mtl_enabled: bool,
    backbone: str,
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
        model, _ = train_mflp_pinn(
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
            mtl_enabled=mtl_enabled,
            backbone=backbone,
        )
        Np_pred_i, Mech_prob_i = predict_outputs(model, Xz[i:i+1], device=device)
        Np_pred_all[i] = float(Np_pred_i[0])
        Mech_prob_all[i] = float(Mech_prob_i[0])
        if (count + 1) % max(1, n // 10) == 0:
            print(f'LOO 进度: {count + 1}/{n}')

    evaluate_and_save_loo(
        material,
        dataset,
        Np_pred_all,
        Mech_prob_all,
        mech_all,
        mtl_enabled,
    )


def parse_hidden_dims(text: str) -> list[int]:
    if text.strip() == '':
        return []
    return [int(x) for x in text.split(',') if x.strip()]


def main(args):

    seed_everything(int(args.seed))
    dev = torch.device('cuda' if (args.device == 'cuda' or (args.device == 'auto' and torch.cuda.is_available())) else 'cpu')
    hidden_dims = parse_hidden_dims(args.hidden_dims)

    include_fp = (not args.no_fp)
    mtl_enabled = (not args.no_mtl)
    backbone = args.backbone.lower()

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
                        mtl_enabled=mtl_enabled,
                        seed=int(args.seed),
                        device=dev,
                        backbone=backbone,
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
                        mtl_enabled=mtl_enabled,
                        test_ratio=float(args.test_ratio),
                        seed=int(args.seed),
                        device=dev,
                        backbone=backbone,
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
                mtl_enabled=mtl_enabled,
                seed=int(args.seed),
                device=dev,
                backbone=backbone,
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
                mtl_enabled=mtl_enabled,
                test_ratio=float(args.test_ratio),
                seed=int(args.seed),
                device=dev,
                backbone=backbone,
            )

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
parser.add_argument('--no-mtl', action='store_true', help='关闭多任务（仅寿命预测）')
parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'])
parser.add_argument('--backbone', type=str, default='transformer', choices=['mlp', 'transformer'], help='选择回骨网络：mlp 或 transformer')

# 你运行哪个就保留哪个args，删掉其他两个
# mlp
args = parser.parse_args(
    ["--epochs", "100", "--backbone", "mlp", "--no-mtl", "--method", "loo"]
)

# transformers
args = parser.parse_args(
    ["--epochs", "100", "--backbone", "transformer", "--no-mtl", "--method", "loo"]
)

# transformers + 多任务学习
args = parser.parse_args(
    ["--epochs", "100", "--backbone", "transformer", "--method", "loo"]
)
main(args)
