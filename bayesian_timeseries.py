import os
import sys
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# 切到脚本所在目录，确保后续的相对路径都正确
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


# 读取/解析函数
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


# 贝叶斯线性回归（NIG 共轭）
def posterior_nig(Phi: np.ndarray, y: np.ndarray, prior_scale2: float = 1e6, alpha0: float = 1e-2, beta0: float = 1e-2):
    n, d = Phi.shape
    m0 = np.zeros(d, dtype=np.float64)
    V0_inv = np.eye(d, dtype=np.float64) / prior_scale2  # V0 = prior_scale2 * I
    A = V0_inv + Phi.T @ Phi  # Vn^{-1}
    b = V0_inv @ m0 + Phi.T @ y
    mn = np.linalg.solve(A, b)
    alpha_n = alpha0 + 0.5 * n
    quad_m0 = m0 @ (V0_inv @ m0)
    quad_mn = mn @ (A @ mn)  # 因为 A = Vn^{-1}
    beta_n = beta0 + 0.5 * (y @ y + quad_m0 - quad_mn)
    return A, mn, alpha_n, beta_n


def predictive_student_t(x: np.ndarray, A: np.ndarray, mn: np.ndarray, alpha_n: float, beta_n: float):
    mu = float(x @ mn)
    Ax = np.linalg.solve(A, x)
    scale2 = float(beta_n / alpha_n) * (1.0 + float(x @ Ax))
    df = 2.0 * alpha_n
    return mu, scale2, df


def t_critical(df: float, level: float = 0.95) -> float:
    # 简单近似：df 大时用正态分布临界值；df 小时使用常见取值
    if df >= 60:
        return 1.959963984540054  # 95%
    table = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
        10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
        19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052,
        28: 2.048, 29: 2.045, 30: 2.042
    }
    k = int(min(max(1, round(df)), 30))
    return table[k]


def loo_predict(Phi: np.ndarray, y: np.ndarray, prior_scale2: float = 1e6, alpha0: float = 1e-2, beta0: float = 1e-2):
    n, d = Phi.shape
    mu_pred = np.zeros(n, dtype=np.float64)
    std_pred = np.zeros(n, dtype=np.float64)
    df_pred = np.zeros(n, dtype=np.float64)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        A, mn, a_n, b_n = posterior_nig(Phi[mask], y[mask], prior_scale2, alpha0, beta0)
        mu_i, s2_i, df_i = predictive_student_t(Phi[i], A, mn, a_n, b_n)
        mu_pred[i] = mu_i
        std_pred[i] = math.sqrt(max(s2_i, 0.0))
        df_pred[i] = df_i
    return mu_pred, std_pred, df_pred


def evaluate_and_save(material: str, dataset: dict, mu_pred: np.ndarray, std_pred: np.ndarray, df_pred: np.ndarray, level: float = 0.95):
    meta = dataset['meta']
    y_true_log = dataset['y']
    y_pred_log = mu_pred
    n = y_true_log.size

    # 置信区间（log 空间）
    ci = np.zeros((n, 2), dtype=np.float64)
    for i in range(n):
        tcrit = t_critical(df_pred[i], level)
        ci[i, 0] = y_pred_log[i] - tcrit * std_pred[i]
        ci[i, 1] = y_pred_log[i] + tcrit * std_pred[i]

    # 映射回原空间
    Nf_true = np.array([m['Nf'] for m in meta], dtype=np.float64)
    Np_pred = 10.0 ** y_pred_log
    Np_lo = 10.0 ** ci[:, 0]
    Np_hi = 10.0 ** ci[:, 1]

    log_err = np.abs(y_pred_log - np.log10(Nf_true))
    mean_log_err = float(np.mean(log_err))
    pct_within_2x = float(np.mean(log_err <= np.log10(2.0)) * 100.0)

    print(f'样本数: {n}, 平均对数误差: {mean_log_err:.4f}, 2倍误差内比例: {pct_within_2x:.2f}%')

    # 保存 CSV
    rows = []
    for i, m in enumerate(meta):
        rows.append({
            'file': m['file'],
            'epsilon_a': m['epsilon_a'],
            'gamma_a': m['gamma_a'],
            'FP': m['FP'],
            'Nf_true': m['Nf'],
            'Np_pred_mean': float(Np_pred[i]),
            'Np_pred_lo': float(Np_lo[i]),
            'Np_pred_hi': float(Np_hi[i]),
            'log_error': float(log_err[i])
        })
    df_out = pd.DataFrame(rows)
    csv_path = get_output_path(material, f'{material}_bayes_timeseries_results.csv')
    df_out.to_csv(csv_path, index=False)
    print(f'结果已保存: {csv_path}')

    # 绘图
    plt.figure(figsize=(8, 8))
    plt.scatter(Nf_true, Np_pred, alpha=0.7, label='Predicted')
    mn = min(float(np.min(Nf_true)), float(np.min(Np_pred)))
    mx = max(float(np.max(Nf_true)), float(np.max(Np_pred)))
    plt.plot([mn, mx], [mn, mx], 'r--', label='Perfect')
    plt.plot([mn, mx], [mn / 2.0, mx / 2.0], 'k:', label='Factor of 2')
    plt.plot([mn, mx], [mn * 2.0, mx * 2.0], 'k:')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('True Life (Nf)')
    plt.ylabel('Predicted Life (Np)')
    plt.title(f'{material} Bayesian Time-series Predictions')
    plt.legend()
    plt.grid(True)
    plot_path = get_output_path(material, f'{material}_bayes_timeseries_scatter.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'散点图已保存: {plot_path}')


def run_material(material: str, include_fp: bool = True, level: float = 0.95):
    print(f'开始处理材料: {material} (Bayesian + 时间序列特征)')
    dataset = build_dataset(material, include_fp=include_fp)
    Phi, y = dataset['Phi'], dataset['y']
    mu_pred_log, std_pred_log, df_pred = loo_predict(Phi, y)
    evaluate_and_save(material, dataset, mu_pred_log, std_pred_log, df_pred, level)


def train_test_split_indices(n: int, test_ratio: float = 0.2, seed: int = 42):
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    t = int(round(n * (1.0 - test_ratio)))
    train_idx = np.sort(idx[:t])
    test_idx = np.sort(idx[t:])
    return train_idx, test_idx


def run_material_split(material: str, include_fp: bool = True, level: float = 0.95, test_ratio: float = 0.2, seed: int = 42):
    print(f'开始处理材料: {material} (Bayesian + 时间序列特征，Train/Test 分色)')
    dataset = build_dataset(material, include_fp=include_fp)
    Phi, y = dataset['Phi'], dataset['y']

    n = Phi.shape[0]
    train_idx, test_idx = train_test_split_indices(n, test_ratio=test_ratio, seed=seed)

    Phi_tr, y_tr = Phi[train_idx], y[train_idx]
    A, mn, a_n, b_n = posterior_nig(Phi_tr, y_tr)

    # 对 train/test 计算后验预测（log 空间）
    mu_tr = np.zeros(train_idx.size, dtype=np.float64)
    s_tr = np.zeros(train_idx.size, dtype=np.float64)
    df_tr = np.zeros(train_idx.size, dtype=np.float64)
    for i, j in enumerate(train_idx):
        mu_i, s2_i, df_i = predictive_student_t(Phi[j], A, mn, a_n, b_n)
        mu_tr[i] = mu_i
        s_tr[i] = math.sqrt(max(s2_i, 0.0))
        df_tr[i] = df_i

    mu_te = np.zeros(test_idx.size, dtype=np.float64)
    s_te = np.zeros(test_idx.size, dtype=np.float64)
    df_te = np.zeros(test_idx.size, dtype=np.float64)
    for i, j in enumerate(test_idx):
        mu_i, s2_i, df_i = predictive_student_t(Phi[j], A, mn, a_n, b_n)
        mu_te[i] = mu_i
        s_te[i] = math.sqrt(max(s2_i, 0.0))
        df_te[i] = df_i

    evaluate_and_save_split(material, dataset, train_idx, test_idx, mu_tr, s_tr, df_tr, mu_te, s_te, df_te, level)


def evaluate_and_save_split(material: str,
                            dataset: dict,
                            train_idx: np.ndarray,
                            test_idx: np.ndarray,
                            mu_tr_log: np.ndarray,
                            s_tr_log: np.ndarray,
                            df_tr: np.ndarray,
                            mu_te_log: np.ndarray,
                            s_te_log: np.ndarray,
                            df_te: np.ndarray,
                            level: float = 0.95):
    meta = dataset['meta']

    # 训练集
    Nf_tr = np.array([meta[i]['Nf'] for i in train_idx], dtype=np.float64)
    Np_tr = 10.0 ** mu_tr_log
    tcrit_tr = np.array([t_critical(df_tr[i], level) for i in range(df_tr.size)], dtype=np.float64)
    Np_tr_lo = 10.0 ** (mu_tr_log - tcrit_tr * s_tr_log)
    Np_tr_hi = 10.0 ** (mu_tr_log + tcrit_tr * s_tr_log)
    log_err_tr = np.abs(mu_tr_log - np.log10(Nf_tr))

    # 测试集
    Nf_te = np.array([meta[i]['Nf'] for i in test_idx], dtype=np.float64)
    Np_te = 10.0 ** mu_te_log
    tcrit_te = np.array([t_critical(df_te[i], level) for i in range(df_te.size)], dtype=np.float64)
    Np_te_lo = 10.0 ** (mu_te_log - tcrit_te * s_te_log)
    Np_te_hi = 10.0 ** (mu_te_log + tcrit_te * s_te_log)
    log_err_te = np.abs(mu_te_log - np.log10(Nf_te))

    print(f'训练集: n={train_idx.size}, 平均对数误差={float(np.mean(log_err_tr)):.4f}')
    print(f'测试集: n={test_idx.size}, 平均对数误差={float(np.mean(log_err_te)):.4f}')

    # 保存 CSV（包含 set 列）
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
            'Np_pred_lo': float(Np_tr_lo[k]),
            'Np_pred_hi': float(Np_tr_hi[k]),
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
            'Np_pred_lo': float(Np_te_lo[k]),
            'Np_pred_hi': float(Np_te_hi[k]),
            'log_error': float(log_err_te[k])
        })
    df_out = pd.DataFrame(rows)
    csv_path = get_output_path(material, f'{material}_bayes_timeseries_results_split.csv')
    df_out.to_csv(csv_path, index=False)
    print(f'分割结果已保存: {csv_path}')

    # 绘图：训练/测试不同颜色（测试点更大更明显）
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
    plt.title(f'{material} Bayesian Predictions (Train/Test)')
    plt.legend()
    plt.grid(True)
    plot_path = get_output_path(material, f'{material}_bayes_timeseries_scatter_split.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'分色散点图已保存: {plot_path}')


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


def main():
    import argparse
    parser = argparse.ArgumentParser(description='时间序列 + 贝叶斯回归（小数据稳健预测）')
    parser.add_argument('--material', type=str, default='ALL', choices=['ALL'] + list(DATA_DIRS.keys()))
    parser.add_argument('--no-fp', action='store_true', help='不使用疲劳参数 FP 作为特征')
    parser.add_argument('--ci', type=float, default=0.95, help='预测区间置信度，例如 0.95')
    parser.add_argument('--method', type=str, default='split', choices=['split', 'loo'], help='预测方式：split 或 loo')
    parser.add_argument('--test-ratio', type=float, default=0.2, help='测试集比例（split 模式）')
    parser.add_argument('--seed', type=int, default=42, help='随机种子（split 模式）')
    args = parser.parse_args()

    if args.material == 'ALL':
        materials = get_available_materials()
        if not materials:
            print('未找到任何材料的应变时间序列数据，无法运行。')
            sys.exit(1)
        print(f"将处理以下材料: {', '.join(materials)}")
        for mat in materials:
            try:
                if args.method == 'loo':
                    run_material(mat, include_fp=(not args.no_fp), level=args.ci)
                else:
                    run_material_split(mat, include_fp=(not args.no_fp), level=args.ci, test_ratio=args.test_ratio, seed=args.seed)
            except Exception as e:
                print(f"处理 {mat} 时发生错误: {e}")
    else:
        if args.method == 'loo':
            run_material(args.material, include_fp=(not args.no_fp), level=args.ci)
        else:
            run_material_split(args.material, include_fp=(not args.no_fp), level=args.ci, test_ratio=args.test_ratio, seed=args.seed)


if __name__ == '__main__':
    main()


