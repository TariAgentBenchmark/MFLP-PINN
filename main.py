import os
import sys

# 切到脚本所在目录，确保后续的相对路径都正确
script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
os.chdir(script_dir)

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
# 把 'TkAgg' → 'Agg'，在无 GUI 的服务器/容器也能正常保存图像
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from tqdm import tqdm  # 添加进度条支持
import math
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.utils.data import DataLoader

# ----------------------------
# 设备配置
# ----------------------------
# 优先使用 GPU，如果有的话
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
if torch.cuda.is_available():
    print(f"GPU 设备名称: {torch.cuda.get_device_name()}")
    print(f"GPU 内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")


# ----------------------------
# 路径配置类
# ----------------------------
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

# 定义数据目录的相对路径
# DATA_DIRS = {
#                'AISI316L': {
#         'strain_series': 'C:/Users/HP/Desktop/AISI316L应变时间序列数据',
#         'fatigue_data': 'C:/Users/HP/Desktop/多轴疲劳试验数据/AISI316L多轴疲劳试验数据.xls'
#                 },
#                 'GH4169': {
#         'strain_series': 'C:/Users/HP/Desktop/GH4169应变时间序列数据',
#         'fatigue_data': 'C:/Users/HP/Desktop/多轴疲劳试验数据/GH4169多轴疲劳试验数据.xls'
#                 },
#                 'TC4': {
#         'strain_series': 'C:/Users/HP/Desktop/TC4应变时间序列数据',
#         'fatigue_data': 'C:/Users/HP/Desktop/多轴疲劳试验数据/TC4多轴疲劳试验数据.xls'
#     }
# }

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


# 输出目录
OUTPUT_DIR = 'results'
    
def get_strain_series_path(material_name):
    """获取应变时间序列数据目录"""
    return DATA_DIRS[material_name]['strain_series']

def get_fatigue_data_path(material_name):
    """获取疲劳数据文件路径"""
    return DATA_DIRS[material_name]['fatigue_data']

def get_output_path(material_name, file_name):
    """获取输出文件路径"""
    # 创建基础输出目录
    base_dir = os.path.join(os.getcwd(), 'results')
    os.makedirs(base_dir, exist_ok=True)
    
    # 创建材料特定的输出目录
    material_dir = os.path.join(base_dir, material_name)
    os.makedirs(material_dir, exist_ok=True)
    
    # 获取完整路径
    full_path = os.path.join(material_dir, file_name)
    
    # 打印路径信息
    print(f"\n保存文件: {file_name}")
    print(f"完整路径: {full_path}")
    
    return full_path

# ----------------------------
# 数据读取和预处理函数
# ----------------------------
def parse_strain_values_from_filename(filename):
    """从文件名中提取应变值对"""
    try:
        parts = filename.replace('strain_series_', '').replace('.xls', '').replace('.csv', '').split('_')
        return float(parts[0]), float(parts[1])
    except Exception as e:
        raise ValueError(f"无法从文件名 {filename} 解析应变值: {str(e)}")

def read_strain_series(file_path):
    """读取应变时间序列数据"""
    try:
        print(f"正在读取时间序列数据: {os.path.basename(file_path)}")
        
        # 检查文件大小
        if os.path.getsize(file_path) == 0:
            raise ValueError("文件为空")
            
        # 根据文件扩展名选择读取方法
        if file_path.endswith('.csv'):
            # 尝试多种编码方式
            encodings = ['utf-8', 'gbk', 'gb2312', 'latin1']
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    print(f"成功使用 {encoding} 编码读取文件")
                    break
                except UnicodeDecodeError:
                    if encoding == encodings[-1]:  # 如果是最后一种编码方式
                        raise
                    continue
        else:
            df = pd.read_excel(file_path)
        
        # 检查必要的列是否存在
        required_columns = ['Time (s)', 'Normal Strain', 'Shear Strain']
        for column in required_columns:
            if column not in df.columns:
                raise ValueError(f"文件缺少必要的列: {column}")
        
        # 确保数据类型为float64
        time = df['Time (s)'].astype(np.float64).values
        normal_strain = df['Normal Strain'].astype(np.float64).values
        shear_strain = df['Shear Strain'].astype(np.float64).values
        
        # 检查数据是否包含无效值
        if np.any(np.isnan(normal_strain)) or np.any(np.isnan(shear_strain)):
            raise ValueError("数据包含NaN值")
        if np.any(np.isinf(normal_strain)) or np.any(np.isinf(shear_strain)):
            raise ValueError("数据包含Inf值")
            
        return time, normal_strain, shear_strain
    except Exception as e:
        raise IOError(f"读取文件 {file_path} 时出错: {str(e)}")

def read_fatigue_data(file_path):
    """读取疲劳试验数据（包含三个表格）"""
    try:
        print(f"正在读取疲劳数据: {os.path.basename(file_path)}")
        
        # 读取所有表格
        all_sheets_data = {
            'epsilon_a': [],  # 正应变幅值
            'gamma_a': [],    # 剪应变幅值
            'Nf': [],         # 真实寿命
            'max_strain': [], # 最大应变
            'max_stress': [], # 最大正应力
            'FP': []          # 疲劳损伤参数
        }
        
        # 读取Excel中的所有表格
        excel_file = pd.ExcelFile(file_path)
        sheet_names = excel_file.sheet_names
        
        print(f"发现表格: {sheet_names}")
        for sheet_name in sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            print(f"读取表格 '{sheet_name}', 包含 {len(df)} 行数据")
            
            # 检查数据列数
            if df.shape[1] < 6:
                print(f"警告: 表格 '{sheet_name}' 列数不足6列，跳过")
                continue
                
            # 添加每个表格的数据
            all_sheets_data['epsilon_a'].extend(df.iloc[:, 0].values)
            all_sheets_data['gamma_a'].extend(df.iloc[:, 1].values)
            all_sheets_data['Nf'].extend(df.iloc[:, 2].values)
            all_sheets_data['max_strain'].extend(df.iloc[:, 3].values)
            all_sheets_data['max_stress'].extend(df.iloc[:, 4].values)
            all_sheets_data['FP'].extend(df.iloc[:, 5].values)
        
        return (np.array(all_sheets_data['epsilon_a']), 
                np.array(all_sheets_data['gamma_a']), 
                np.array(all_sheets_data['Nf']), 
                np.array(all_sheets_data['max_strain']), 
                np.array(all_sheets_data['max_stress']), 
                np.array(all_sheets_data['FP']))
    except Exception as e:
        raise IOError(f"读取疲劳数据文件 {file_path} 时出错: {str(e)}")


# ----------------------------
# 定义 Transformer + 注意力机制 提取动态损伤参量
# ----------------------------
class TransformerFeatureExtractor(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super(TransformerFeatureExtractor, self).__init__()
        
        self.d_model = d_model
        
        # 输入映射层：将输入特征维度映射到transformer所需的维度
        self.input_proj = nn.Linear(input_size, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True  # 使用batch_first=True以匹配输入格式
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # 确保输入维度正确
        if len(x.shape) == 2:
            x = x.unsqueeze(0)  # 添加batch维度 [1, seq_len, features]
        
        # 输入映射
        x = self.input_proj(x)  # [batch_size, seq_len, d_model]
        
        # 添加位置编码
        x = self.pos_encoder(x)
        
        # Transformer编码器
        transformer_output = self.transformer_encoder(x)  # [batch_size, seq_len, d_model]
        
        # 使用最后一个时间步的输出作为特征
        last_hidden = transformer_output[:, -1, :]  # [batch_size, d_model]
        
        return last_hidden

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.transpose(0, 1)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# ----------------------------
# 定义物理损失函数
# ----------------------------
def fs_loss(Np_pred, FP, material_name, eps=1e-6):
    """计算FS准则物理损失（根据材料参数），并保证不会出现 log10(0) 的 nan"""
    params = materials_params[material_name]
    tau_f_prime   = torch.tensor(params['tau_f_G'], device=Np_pred.device)
    gamma_f_prime = torch.tensor(params['gamma_f'],   device=Np_pred.device)
    b_o           = torch.tensor(params['b0'],        device=Np_pred.device)
    c_o           = torch.tensor(params['c0'],        device=Np_pred.device)
    G             = torch.tensor(params['G'],         device=Np_pred.device)

    # 1) clamp 预测寿命到合理范围，避免 0 或负数
    Np_safe = torch.clamp(Np_pred, min=eps, max=1e5)

    # 2) 按 FS 理论公式计算预测 FP
    term1 = (tau_f_prime / G) * torch.pow(2 * Np_safe, b_o)
    term2 = gamma_f_prime * torch.pow(2 * Np_safe, c_o)
    FP_pred = term1 + term2

    # 3) clamp 真 FP，避免 ≤0 导致 log10 NaN
    FP_pos = torch.clamp(FP, min=eps)

    # 4) 对数变换前再加 eps
    log_FP_pred = torch.log10(FP_pred + eps)
    log_FP      = torch.log10(FP_pos   + eps)

    # 5) 计算相对误差并用 Huber 损失
    relative_error = torch.abs(log_FP_pred - log_FP) / (torch.abs(log_FP) + eps)
    delta = 0.1
    huber_mask = relative_error < delta
    loss = torch.where(
        huber_mask,
        0.5 * relative_error ** 2,
        delta * relative_error - 0.5 * delta ** 2
    )

    return torch.mean(loss)



def rmse_loss(Nf, Np_pred):
    """计算RMSE损失，在对数空间中计算"""
    # 确保预测值在合理范围内
    Np_pred = torch.clamp(Np_pred, min=100.0, max=1e7)
    Nf = torch.clamp(Nf, min=100.0)
    
    # 在对数空间中计算MSE
    log_mse = torch.mean((torch.log10(Np_pred) - torch.log10(Nf)) ** 2)
    
    # 计算相对误差作为辅助损失
    relative_error = torch.mean(torch.abs(Np_pred - Nf) / Nf)
    
    # 组合损失
    return log_mse + 0.1 * relative_error

# === Soft-label 二分类交叉熵损失函数 === #
def soft_binary_cross_entropy(pred, target, eps=1e-6):
    """
    pred: sigmoid 之后的概率
    target: [0,1] 范围内的软标签
    """
    # 强制 clamp 到 (eps, 1-eps)
    pred = torch.clamp(pred, min=eps, max=1.0 - eps)
    return F.binary_cross_entropy(pred, target)


# === 多任务总损失函数（支持动态归一化） === #
class LossNormalizer:
    def __init__(self, momentum=0.99, eps=1e-8):
        self.momentum = momentum
        self.eps = eps
        self.mean = None

    def update(self, value: torch.Tensor):
        # 只用 value.item() 更新 self.mean，但不把 value 本身转为 float
        v = value.item()
        if self.mean is None:
            self.mean = v
        else:
            self.mean = self.momentum * self.mean + (1 - self.momentum) * v
        return self.mean

    def norm(self, value: torch.Tensor):
        # 直接用 Tensor (value) / float(self.mean + eps)，返回仍然是 Tensor
        return value / (self.mean + self.eps)

def multitask_total_loss(pred_life, true_life, pred_mech, true_mech, physical_loss,
                         alpha=1.0, beta=1.0, gamma=0.3,
                         normers=None):
    # 1) 寿命分支 loss
    loss_life = rmse_loss(true_life, pred_life)

    # 2) 机制分支 loss
    valid = (true_mech >= 0)
    if valid.sum() > 0:
        loss_mech = soft_binary_cross_entropy(pred_mech[valid], true_mech[valid])
    else:
        # 用零张量代替 Python float
        loss_mech = torch.tensor(0.0, device=pred_mech.device)

    # 3) physical_loss 已经是 tensor

    # 4) 如果要做动态归一化
    if normers is not None:
        # 先用 tensor loss_life.loss_phys.loss_mech 更新滑动平均
        normers['life'].update(loss_life)
        normers['phys'].update(physical_loss)
        normers['mech'].update(loss_mech)

        # 再得到张量形式的归一化 loss
        nl = normers['life'].norm(loss_life)
        np = normers['phys'].norm(physical_loss)
        nm = normers['mech'].norm(loss_mech)

        total = alpha * nl + beta * np + gamma * nm
    else:
        total = alpha * loss_life + beta * physical_loss + gamma * loss_mech

    return total, loss_life, physical_loss, loss_mech


def multitask_total_loss(pred_life, true_life, pred_mech, true_mech, physical_loss,
                         alpha=1.0, beta=1.0, gamma=0.3,
                         normers=None):
    # 1) 寿命分支 loss
    loss_life = rmse_loss(true_life, pred_life)

    # 2) 机制分支 loss
    valid = (true_mech >= 0)
    if valid.sum() > 0:
        loss_mech = soft_binary_cross_entropy(pred_mech[valid], true_mech[valid])
    else:
        # 用零张量代替 Python float
        loss_mech = torch.tensor(0.0, device=pred_mech.device)

    # 3) physical_loss 已经是 tensor

    # 4) 如果要做动态归一化
    if normers is not None:
        # 先用 tensor loss_life.loss_phys.loss_mech 更新滑动平均
        normers['life'].update(loss_life)
        normers['phys'].update(physical_loss)
        normers['mech'].update(loss_mech)

        # 再得到张量形式的归一化 loss
        nl = normers['life'].norm(loss_life)
        np = normers['phys'].norm(physical_loss)
        nm = normers['mech'].norm(loss_mech)

        total = alpha * nl + beta * np + gamma * nm
    else:
        total = alpha * loss_life + beta * physical_loss + gamma * loss_mech

    return total, loss_life, physical_loss, loss_mech
# ----------------------------
# 定义 MFLP-PINN 神经网络结构
# ----------------------------
class MFLP_PINN(nn.Module):
    def __init__(self, input_size, hidden_layers=[128, 256, 128, 64], dropout=0.2):
        super(MFLP_PINN, self).__init__()
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout
        layers = []
        prev_size = input_size
        for size in hidden_layers:
            layers.append(nn.Linear(prev_size, size))
            nn.init.kaiming_normal_(layers[-1].weight, nonlinearity='relu')
            nn.init.zeros_(layers[-1].bias)
            layers.append(nn.LayerNorm(size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = size
        self.backbone = nn.Sequential(*layers)
        self.life_branch = nn.Sequential(
            nn.Linear(prev_size, prev_size // 2),
            nn.LayerNorm(prev_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(prev_size // 2, 1)
        )
        for m in self.life_branch.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)
        # 机制分类分支
        self.mechanism_head = nn.Linear(prev_size, 1)
        nn.init.kaiming_normal_(self.mechanism_head.weight, nonlinearity='sigmoid')
        nn.init.zeros_(self.mechanism_head.bias)
        self.min_life = 100.0
        self.max_life = 1e7
        self.log_min_life = np.log10(self.min_life)
        self.log_max_life = np.log10(self.max_life)
    def forward(self, x):
        features = self.backbone(x)
        log_life = self.life_branch(features).squeeze(-1)  # 直接输出 log10(Np)
        life = torch.pow(10.0, log_life)
        mech_prob = torch.sigmoid(self.mechanism_head(features))
        return life, mech_prob

# ----------------------------
# 数据集类
# ----------------------------

class FatigueDataset(torch.utils.data.Dataset):
    def __init__(self,
                     material_name='AISI316L',
                     split='train',
                     train_ratio=0.7,
                     val_ratio=0.15,
                     test_ratio=0.15,
                     seed=42,
                     eps_mean=None, eps_std=None,
                     gam_mean=None, gam_std=None,
                     FP_mean=None, FP_std=None):

            super(FatigueDataset, self).__init__()
            torch.manual_seed(seed)
            np.random.seed(seed)

            # 路径字典
            self.paths = {
                'fatigue_data_dir': None,
                'data_dirs': DATA_DIRS.copy()
            }

            # 存储归一化参数
            self.eps_mean, self.eps_std = eps_mean, eps_std
            self.gam_mean, self.gam_std = gam_mean, gam_std
            self.FP_mean, self.FP_std = FP_mean, FP_std

            # 获取应变序列路径
            strain_series_path = get_strain_series_path(material_name)
            if not os.path.exists(strain_series_path):
                self.scan_for_paths()
                strain_series_path = get_strain_series_path(material_name)
                if not os.path.exists(strain_series_path):
                    raise FileNotFoundError(f"找不到应变时间序列文件夹: {strain_series_path}")

            # 扫描所有文件对
            valid_pairs = []
            for fname in os.listdir(strain_series_path):
                if fname.startswith('strain_series_') and (fname.endswith('.xls') or fname.endswith('.csv')):
                    full_path = os.path.join(strain_series_path, fname)
                    try:
                        ea, ga = parse_strain_values_from_filename(fname)
                        valid_pairs.append((full_path, ea, ga))
                    except:
                        continue
            if not valid_pairs:
                raise ValueError(f"没有找到任何有效的{material_name}数据对！")

            # 划分 train/val/test
            np.random.shuffle(valid_pairs)
            n = len(valid_pairs)
            t_end = int(n * train_ratio)
            v_end = t_end + int(n * val_ratio)
            if split == 'train':
                pairs = valid_pairs[:t_end]
            elif split == 'val':
                pairs = valid_pairs[t_end:v_end]
            elif split == 'test':
                pairs = valid_pairs[v_end:]
            else:
                raise ValueError(f"无效的 split 参数: {split}")

            # 读取疲劳实验数据
            fatigue_path = get_fatigue_data_path(material_name)
            if not os.path.exists(fatigue_path):
                raise FileNotFoundError(f"找不到疲劳数据文件: {fatigue_path}")
            eps_all, gam_all, Nf_all, _, _, FP_all = read_fatigue_data(fatigue_path)

            # 匹配每一对序列和疲劳数据
            self.matched_data = []
            for seq_path, ea, ga in pairs:
                mask = (np.isclose(eps_all, ea, atol=1e-5) & np.isclose(gam_all, ga, atol=1e-5))
                idxs = np.where(mask)[0]
                if idxs.size == 0:
                    continue
                idx = idxs[0]
                try:
                    t, ns, ss = read_strain_series(seq_path)
                    series = np.stack([ns, ss], axis=1)
                    self.matched_data.append({
                        'strain_series': series,
                        'epsilon_a': ea,
                        'gamma_a': ga,
                        'Nf': Nf_all[idx],
                        'FP': float(FP_all[idx])
                    })
                except:
                    continue
            if not self.matched_data:
                raise ValueError(f"没有任何匹配的 {material_name} 数据！")

            # 转为张量
            # print(self.matched_data)

            # for d in self.matched_data:
            #     for key in ['gamma_a', 'epsilon_a', 'Nf']:
            #         if isinstance(d[key], str):
            #             print(f"类型{key}：", {d[key]})

        
            arr = np.array([d['strain_series'] for d in self.matched_data], dtype=np.float32)
            self.strain_series = torch.from_numpy(arr)
            self.epsilon_a = torch.tensor([d['epsilon_a'] for d in self.matched_data], dtype=torch.float32).unsqueeze(1)
            self.gamma_a = torch.tensor([d['gamma_a'] for d in self.matched_data], dtype=torch.float32).unsqueeze(1)
            self.Nf = torch.tensor([d['Nf'] for d in self.matched_data], dtype=torch.float32).unsqueeze(1)
            self.FP = torch.tensor([d['FP'] for d in self.matched_data], dtype=torch.float32).unsqueeze(1)

            # 机制 soft-label
            def gen_label(mat, ga, ea):
                R = ga / ea
                if mat == 'TC4':
                    return 1.0 if R < 1.4 else (0.6 if R < 1.6 else 0.0)
                if mat == 'GH4169':
                    return 1.0 if R < 1.35 else (0.5 if R < 1.55 else 0.0)
                return -1.0

            if material_name in ('TC4', 'GH4169'):
                labels = [gen_label(material_name, d['gamma_a'], d['epsilon_a']) for d in self.matched_data]
                self.mech_label = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
            else:
                self.mech_label = torch.full((len(self.matched_data), 1), -1.0, dtype=torch.float32)

    def __len__(self):
        return len(self.matched_data)

    def __getitem__(self, idx):
        seq = self.strain_series[idx]
        eps = self.epsilon_a[idx]
        gam = self.gamma_a[idx]
        Nf  = self.Nf[idx]
        FPv = self.FP[idx]
        mech = self.mech_label[idx]

        # Z-score 标准化
        eps_n = (eps - self.eps_mean) / (self.eps_std + 1e-6)
        gam_n = (gam - self.gam_mean) / (self.gam_std + 1e-6)
        FP_n  = (FPv - self.FP_mean)  / (self.FP_std  + 1e-6)

        return seq, eps_n, gam_n, Nf, FP_n, mech

    def scan_for_paths(self):
        """扫描常见位置查找数据文件夹"""
        # 常见位置列表
        common_locations = [
            os.getcwd(),  # 当前工作目录
            os.path.expanduser("~"),  # 用户主目录
            os.path.join(os.path.expanduser("~"), "Desktop"),  # 桌面
            os.path.join(os.path.expanduser("~"), "Documents"),  # 文档
            "D:\\",  # D盘根目录
            "E:\\",  # E盘根目录
        ]
        
        # 查找多轴疲劳试验数据文件夹
        for location in common_locations:
            if os.path.exists(location):
                fatigue_data_dir = os.path.join(location, "多轴疲劳试验数据")
                if os.path.exists(fatigue_data_dir):
                    self.paths['fatigue_data_dir'] = fatigue_data_dir
                    print(f"找到疲劳数据目录: {fatigue_data_dir}")
                
                # 查找各材料的应变时间序列数据
                for material in self.paths['data_dirs'].keys():
                    strain_dir = os.path.join(location, f"{material}应变时间序列数据")
                    if os.path.exists(strain_dir):
                        self.paths['data_dirs'][material]['strain_series'] = strain_dir
                        print(f"找到{material}应变时间序列目录: {strain_dir}")
        
        # 保存扫描结果
        self.save_config()

    def save_config(self):
        # 如果你暂时不需要持久化，可以先空实现，防止报错
        pass

# ----------------------------
# 模型训练函数
# ----------------------------
def train_model(material_name):
    print(f"\n开始训练{material_name}材料的疲劳寿命预测模型...")

    # —— 临时加载 train split，计算三个特征的均值和标准差 ——
    # 先传入任意占位的归一化参数，真正的 eps_mean 等会马上被覆盖
    tmp = FatigueDataset(
        material_name, split='train',
        eps_mean=0, eps_std=1,
        gam_mean=0, gam_std=1,
        FP_mean=0,  FP_std=1
    )
    eps_mean, eps_std = tmp.epsilon_a.mean(), tmp.epsilon_a.std(unbiased=False)
    gam_mean, gam_std = tmp.gamma_a.mean(), tmp.gamma_a.std(unbiased=False)
    FP_mean,  FP_std  = tmp.FP.mean(),      tmp.FP.std(unbiased=False)

    # —— 第二步：带归一化参数初始化数据集 ——
    train_dataset = FatigueDataset(
        material_name, split='train',
        eps_mean=eps_mean, eps_std=eps_std,
        gam_mean=gam_mean, gam_std=gam_std,
        FP_mean=FP_mean, FP_std=FP_std
    )
    val_dataset = FatigueDataset(
        material_name, split='val',
        eps_mean=eps_mean, eps_std=eps_std,
        gam_mean=gam_mean, gam_std=gam_std,
        FP_mean=FP_mean, FP_std=FP_std
    )
    test_dataset = FatigueDataset(
        material_name, split='test',
        eps_mean=eps_mean, eps_std=eps_std,
        gam_mean=gam_mean, gam_std=gam_std,
        FP_mean=FP_mean, FP_std=FP_std
    )

    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader   = torch.utils.data.DataLoader(val_dataset,   batch_size=16, shuffle=False)
    test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=4, shuffle=False)

    # 初始化模型
    input_size = 2  # 正应变和剪应变
    d_model    = 64   # Transformer隐藏维度
    nhead      = 4    # 注意力头数
    num_layers = 2    # 编码层数
    transformer_model = TransformerFeatureExtractor(input_size, d_model, nhead, num_layers, dropout=0.1).to(device)

    pinn_input_size = d_model + 3  # Transformer特征 + εₐ + γₐ + FP
    hidden_layers   = [64, 128, 64, 32]
    pinn_model      = MFLP_PINN(pinn_input_size, hidden_layers, dropout=0.1).to(device)

    # 优化器 & 学习率调度
    optimizer = optim.AdamW(
        list(transformer_model.parameters()) + list(pinn_model.parameters()),
        lr=0.001, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=50, min_lr=1e-6
    )

    # 训练超参
    num_epochs      = 2000
    lambda_data     = 1.0
    lambda_physical = 0.1
    patience        = 200
    early_stop_ctr  = 0
    best_loss       = float('inf')
    best_epoch      = 0
    best_t_state    = None
    best_p_state    = None

    loss_history = {
        "train_loss": [], "train_data_loss": [], "train_physical_loss": [], "train_mech_loss": [],
        "val_loss":   [], "val_data_loss":   [], "val_physical_loss":   [], "val_mech_loss":   []
    }

    # —— 新增：动态归一化器 ——
    normers = {
        'life': LossNormalizer(),
        'phys': LossNormalizer(),
        'mech': LossNormalizer()
    }

    plt.ion()
    fig = plt.figure(figsize=(15, 10))

    print(f"\n开始训练模型 (共 {num_epochs} 轮)...")
    for epoch in tqdm(range(num_epochs)):
        # ========== 训练阶段 ==========
        transformer_model.train()
        pinn_model.train()
        t_loss = d_loss = p_loss = m_loss = 0.0

        for X_seq, eps, gam, Nf, FPv, mech_lbl in train_loader:
            # 移动数据到指定设备
            X_seq = X_seq.to(device)
            eps = eps.to(device)
            gam = gam.to(device)
            Nf = Nf.to(device)
            FPv = FPv.to(device)
            mech_lbl = mech_lbl.to(device)

            optimizer.zero_grad()
            feats = transformer_model(X_seq)
            Xc    = torch.cat([feats, eps, gam, FPv], dim=1)
            pred_life, pred_mech = pinn_model(Xc)

            loss_data = rmse_loss(Nf, pred_life)
            loss_phys = fs_loss(pred_life, FPv, material_name)
            # train／val 循环里
            if (mech_lbl >= 0).sum() > 0:
                # 先 clamp 到 (eps, 1-eps)，防止 0/1 出现
                probs = pred_mech[mech_lbl >= 0].clamp(1e-6, 1 - 1e-6)
                loss_mech = soft_binary_cross_entropy(
                    probs,
                    mech_lbl[mech_lbl >= 0]
                )
            else:
                loss_mech = 0.0

            multitask, ld, lp, lm = multitask_total_loss(
                pred_life, Nf, pred_mech, mech_lbl, loss_phys,
                alpha=lambda_data, beta=lambda_physical, gamma=0.3, normers=normers
            )
            multitask.backward()
            torch.nn.utils.clip_grad_norm_(transformer_model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(pinn_model.parameters(), max_norm=1.0)
            optimizer.step()

            t_loss += multitask.item()
            d_loss += ld.item()
            p_loss += lp.item()
            m_loss += lm.item() if hasattr(lm, 'item') else lm

        # ========== 验证阶段 ==========
        transformer_model.eval()
        pinn_model.eval()
        v_loss = vd_loss = vp_loss = vm_loss = 0.0
        with torch.no_grad():
            for X_seq, eps, gam, Nf, FPv, mech_lbl in val_loader:
                # 移动数据到指定设备
                X_seq = X_seq.to(device)
                eps = eps.to(device)
                gam = gam.to(device)
                Nf = Nf.to(device)
                FPv = FPv.to(device)
                mech_lbl = mech_lbl.to(device)

                feats = transformer_model(X_seq)
                Xc    = torch.cat([feats, eps, gam, FPv], dim=1)
                pred_life, pred_mech = pinn_model(Xc)

                ld = rmse_loss(Nf, pred_life)
                lp = fs_loss(pred_life, FPv, material_name)
                if (mech_lbl >= 0).sum() > 0:
                    lm = soft_binary_cross_entropy(
                        pred_mech[mech_lbl >= 0], mech_lbl[mech_lbl >= 0]
                    )
                else:
                    lm = 0.0

                multitask, _, _, _ = multitask_total_loss(
                    pred_life, Nf, pred_mech, mech_lbl, lp,
                    alpha=lambda_data, beta=lambda_physical, gamma=0.3, normers=None
                )
                v_loss  += multitask.item()
                vd_loss += ld.item()
                vp_loss += lp.item()
                vm_loss += lm.item() if hasattr(lm, 'item') else lm

        # ========== 记录 & 早停 ==========
        avg_t  = t_loss / len(train_loader)
        avg_v  = v_loss / len(val_loader)
        avg_td = d_loss / len(train_loader)
        avg_vd = vd_loss / len(val_loader)
        avg_tp = p_loss / len(train_loader)
        avg_vp = vp_loss / len(val_loader)
        avg_tm = m_loss / len(train_loader)
        avg_vm = vm_loss / len(val_loader)

        scheduler.step(avg_v)
        loss_history["train_loss"].append(avg_t)
        loss_history["val_loss"].append(avg_v)
        loss_history["train_data_loss"].append(avg_td)
        loss_history["val_data_loss"].append(avg_vd)
        loss_history["train_physical_loss"].append(avg_tp)
        loss_history["val_physical_loss"].append(avg_vp)
        loss_history["train_mech_loss"].append(avg_tm)
        loss_history["val_mech_loss"].append(avg_vm)

        if avg_v < best_loss:
            best_loss       = avg_v
            best_epoch      = epoch
            best_t_state    = transformer_model.state_dict().copy()
            best_p_state    = pinn_model.state_dict().copy()
            early_stop_ctr  = 0
        else:
            early_stop_ctr += 1
            if early_stop_ctr >= patience:
                print(f"早停: {epoch+1}轮后没有改善")
                break

        # ========== 可视化 & 日志（每 10 轮）==========
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"\nEpoch [{epoch+1}/{num_epochs}]")
            print(f"Train Loss: {avg_t:.4f}, Val Loss: {avg_v:.4f}")
            print(f"Train Data Loss: {avg_td:.4f}, Val Data Loss: {avg_vd:.4f}")
            print(f"Train Physical Loss: {avg_tp:.4f}, Val Physical Loss: {avg_vp:.4f}")
            print(f"Train Mech Loss: {avg_tm:.4f}, Val Mech Loss: {avg_vm:.4f}")
            print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

            plt.clf()
            # 子图1: 损失历史
            plt.subplot(2, 2, 1)
            plt.plot(loss_history["train_loss"], 'b-', label='Train Loss')
            plt.plot(loss_history["val_loss"], 'r-', label='Val Loss')
            plt.plot(loss_history["train_data_loss"], 'b--', label='Train Data Loss')
            plt.plot(loss_history["val_data_loss"], 'r--', label='Val Data Loss')
            plt.plot(loss_history["train_physical_loss"], 'b:', label='Train Physical Loss')
            plt.plot(loss_history["val_physical_loss"], 'r:', label='Val Physical Loss')
            plt.plot(loss_history["train_mech_loss"], 'g--', label='Train Mech Loss')
            plt.plot(loss_history["val_mech_loss"], 'g:', label='Val Mech Loss')
            plt.yscale('log')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            plt.title('Training & Validation Loss')

            # ---------------- 子图2: 预测 vs 真实 ----------------
            plt.subplot(2, 2, 2)
            transformer_model.eval();
            pinn_model.eval()
            with torch.no_grad():
                feats = transformer_model(val_dataset.strain_series)
                Xc = torch.cat([feats, val_dataset.epsilon_a, val_dataset.gamma_a, val_dataset.FP], dim=1)
                preds, _ = pinn_model(Xc)

            # detach & cpu -> numpy
            true_vals = val_dataset.Nf.detach().cpu().numpy()
            preds_np = preds.detach().cpu().numpy()

            mn = float(min(true_vals.min(), preds_np.min()))
            mx = float(max(true_vals.max(), preds_np.max()))

            plt.scatter(val_dataset.Nf.detach().cpu().numpy(),
                        preds.detach().cpu().numpy(),
                        c='b', label='Predictions')

            plt.plot([mn, mx], [mn, mx], 'r--', label='Perfect')
            plt.plot([mn, mx], [mn / 2, mx / 2], 'k:', label='×1.5')
            plt.plot([mn, mx], [mn * 2, mx * 2], 'k:')
            plt.xscale('log');
            plt.yscale('log')
            plt.xlabel('True Life');
            plt.ylabel('Pred Life')
            plt.legend();
            plt.grid(True)
            transformer_model.train();
            pinn_model.train()

            # 子图3: 学习率
            plt.subplot(2, 2, 3)
            lr = optimizer.param_groups[0]['lr']
            plt.scatter(epoch, lr, c='b')
            plt.yscale('log')
            plt.xlabel('Epoch'); plt.ylabel('LR')
            plt.grid(True)
            plt.title('Learning Rate')

            # ---------------- 子图4: 对数误差分布 ----------------
            plt.subplot(2, 2, 4)
            # 计算对数误差并 detach、cpu、flatten
            errs = torch.abs(torch.log10(preds) - torch.log10(val_dataset.Nf))
            errs_np = errs.detach().cpu().numpy().flatten()

            # 只传入一维数组，去掉多余的 color 参数
            plt.hist(errs_np, bins=20, alpha=0.7)
            plt.axvline(x=np.log10(2), color='r', linestyle='--', label='×1.5')
            plt.xlabel('Log Error')
            plt.ylabel('Count')
            plt.legend()
            plt.grid(True)
            plt.title('Log Error Dist')

            plt.ioff(); plt.close()

    # 恢复最佳模型
    if best_t_state is not None:
        transformer_model.load_state_dict(best_t_state)
        pinn_model.load_state_dict(best_p_state)
    print(f'训练完成。加载最佳模型 (Epoch {best_epoch+1}, Val Loss={best_loss:.4f})')

    # ========== 测试集评估 ==========
    print("\n在测试集上评估最终模型...")
    transformer_model.eval(); pinn_model.eval()
    test_loss = test_data_loss = test_physical_loss = 0.0
    all_predictions, all_true_values = [], []
    all_pred_mech, all_true_mech = [], []

    with torch.no_grad():
        for X_seq, eps, gam, Nf, FPv, mech_lbl in test_loader:
            # 移动数据到指定设备
            X_seq = X_seq.to(device)
            eps = eps.to(device)
            gam = gam.to(device)
            Nf = Nf.to(device)
            FPv = FPv.to(device)
            mech_lbl = mech_lbl.to(device)

            feats = transformer_model(X_seq)
            Xc    = torch.cat([feats, eps, gam, FPv], dim=1)
            pred_life, pred_mech = pinn_model(Xc)

            ld = rmse_loss(Nf, pred_life)
            lp = fs_loss(pred_life, FPv, material_name)
            if (mech_lbl >= 0).sum() > 0:
                lm = soft_binary_cross_entropy(pred_mech[mech_lbl >= 0], mech_lbl[mech_lbl >= 0])
            else:
                lm = 0.0

            loss = lambda_data * ld + lambda_physical * lp + 0.3 * lm
            test_loss += loss.item()
            test_data_loss += ld.item()
            test_physical_loss += lp.item()

            all_predictions.extend(pred_life.cpu().numpy())
            all_true_values.extend(Nf.cpu().numpy())
            all_pred_mech.extend(pred_mech.cpu().numpy())
            all_true_mech.extend(mech_lbl.cpu().numpy())

    avg_test_loss      = test_loss / len(test_loader)
    avg_test_data_loss = test_data_loss / len(test_loader)
    avg_test_phys_loss = test_physical_loss / len(test_loader)
    print(f"\n测试集结果: Test Loss={avg_test_loss:.4f}, Data Loss={avg_test_data_loss:.4f}, Physical Loss={avg_test_phys_loss:.4f}")

    # 保存预测结果 CSV
    results_path = get_output_path(material_name, f"{material_name}_test_predictions.csv")
    results_df = pd.DataFrame({
        'True_Life': np.array(all_true_values).flatten(),
        'Predicted_Life': np.array(all_predictions).flatten(),
        'True_Mechanism': np.array(all_true_mech).flatten(),
        'Predicted_Mechanism_Prob': np.array(all_pred_mech).flatten()
    })
    results_df.to_csv(results_path, index=False)
    print(f"预测结果已保存到: {results_path}")

    # 计算 AUC 和 准确率
    valid_mask = np.array(all_true_mech).flatten() >= 0
    if valid_mask.sum() > 0:
        true_bin = (np.array(all_true_mech)[valid_mask] >= 0.5).astype(int)
        prob     = np.array(all_pred_mech)[valid_mask]
        auc = roc_auc_score(true_bin, prob)
        acc = accuracy_score(true_bin, (prob >= 0.5).astype(int))
        print(f"机制预测 AUC: {auc:.4f}, 准确率: {acc:.4f}")

    # 绘制并保存测试散点图
    true_vals = np.array(all_true_values).flatten()
    preds_np = np.array(all_predictions).flatten()
    plt.figure(figsize=(10, 8))
    plt.scatter(true_vals, preds_np, alpha=0.6, label='预测值')
    mn = float(min(true_vals.min(), preds_np.min()))
    mx = float(max(true_vals.max(), preds_np.max()))
    plt.plot([mn, mx], [mn, mx], 'r--', label='完美预测')
    plt.plot([mn, mx], [mn/2, mx/2], 'k:', label='×1.5')
    plt.plot([mn, mx], [mn*2, mx*2], 'k:')
    plt.xscale('log'); plt.yscale('log')
    plt.xlabel('真实疲劳寿命 (Nf)'); plt.ylabel('预测疲劳寿命 (Np)')
    plt.title(f'{material_name} 测试集预测结果')
    plt.legend(); plt.grid(True)
    plot_path = get_output_path(material_name, f"{material_name}_test_prediction_plot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"测试集预测图已保存到: {plot_path}")

    return transformer_model, pinn_model, loss_history, test_dataset


# ----------------------------
# 留一交叉验证
# ----------------------------
def train_with_leave_one_out(transformer_model, pinn_model, dataset, material_name):
    """针对小数据集使用留一交叉验证训练模型"""
    print(f"\n使用留一交叉验证训练{material_name}模型...")

    n_samples = len(dataset)
    all_pred_lifes = []
    all_true_lifes = []
    all_strain_names = []
    all_epsilon_a_values = []
    all_gamma_a_values = []
    all_fp_values = []

    # 保存原始数据，用于后续可视化
    all_strain_series = dataset.strain_series  # shape (N, seq_len, 2)
    all_epsilon_a = dataset.epsilon_a          # shape (N,1)
    all_gamma_a = dataset.gamma_a              # shape (N,1)
    all_Nf = dataset.Nf                        # shape (N,1)
    all_FP = dataset.FP                        # shape (N,1)

    # 获取 strain_series_name 列表
    strain_series_names = [f"strain_series_{i+1}" for i in range(n_samples)]

    # 创建输出目录
    output_dir = get_output_path(material_name, "")
    os.makedirs(output_dir, exist_ok=True)

    for test_idx in range(n_samples):
        print(f"\n--- 留一交叉验证 - 测试样本 {test_idx+1}/{n_samples} ---")

        # 划分训练/测试
        train_indices = [i for i in range(n_samples) if i != test_idx]
        test_index = test_idx

        train_strain = all_strain_series[train_indices].to(device)  # (N-1, seq_len, 2)
        train_epsilon_a = all_epsilon_a[train_indices].to(device)   # (N-1,1)
        train_gamma_a = all_gamma_a[train_indices].to(device)
        train_Nf = all_Nf[train_indices].to(device)
        train_FP = all_FP[train_indices].to(device)

        test_strain = all_strain_series[test_index].unsqueeze(0).to(device)      # (1, seq_len, 2)
        test_epsilon_a = all_epsilon_a[test_index].unsqueeze(0).to(device)      # (1,1)
        test_gamma_a = all_gamma_a[test_index].unsqueeze(0).to(device)
        test_Nf = all_Nf[test_index].unsqueeze(0).to(device)
        test_FP = all_FP[test_index].unsqueeze(0).to(device)

        # 重新初始化模型（每折独立）
        transformer_input_size = 2
        d_model = 32
        nhead = 2
        num_layers = 1
        transformer_model_fold = TransformerFeatureExtractor(transformer_input_size, d_model, nhead, num_layers, dropout=0.0).to(device)
        pinn_input_size = d_model + 3
        hidden_layers = [32, 16]
        pinn_model_fold = MFLP_PINN(pinn_input_size, hidden_layers, dropout=0.0).to(device)

        optimizer = optim.AdamW(
            list(transformer_model_fold.parameters()) + list(pinn_model_fold.parameters()),
            lr=0.002,
            weight_decay=1e-5
        )

        num_epochs = 200
        lambda_data = 1.0
        lambda_physical = 0.1

        best_loss = float('inf')
        patience = 30
        early_stop_counter = 0
        min_epochs = 50

        loss_history = {
            "epochs": [],
            "total_loss": [],
            "data_loss": [],
            "physical_loss": [],
            "log_error": []
        }

        for epoch in range(num_epochs):
            transformer_model_fold.train()
            pinn_model_fold.train()

            # 前向（训练集）：train_strain 形状是 (N-1, seq_len, 2)，已经有 batch 维
            transformer_features = transformer_model_fold(train_strain)  # (N-1, d_model)
            X_combined = torch.cat([transformer_features, train_epsilon_a, train_gamma_a, train_FP], dim=1)  # (N-1, d_model+3)

            Np_pred, _ = pinn_model_fold(X_combined)  # (N-1, )
            loss_data = rmse_loss(train_Nf, Np_pred.unsqueeze(1) if Np_pred.dim()==1 else Np_pred)  # 保持 shape 匹配
            loss_physical = fs_loss(Np_pred.unsqueeze(1) if Np_pred.dim()==1 else Np_pred, train_FP, material_name)
            loss = lambda_data * loss_data + lambda_physical * loss_physical

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(transformer_model_fold.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(pinn_model_fold.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss = loss.item()
            epoch_data_loss = loss_data.item()
            epoch_physical_loss = loss_physical.item()

            # 评估当前测试样本的误差
            transformer_model_fold.eval()
            pinn_model_fold.eval()
            with torch.no_grad():
                test_features = transformer_model_fold(test_strain)  # (1, d_model)
                test_combined = torch.cat([
                    test_features,
                    test_epsilon_a,
                    test_gamma_a,
                    test_FP
                ], dim=1)  # (1, d_model+3)

                test_pred, _ = pinn_model_fold(test_combined)  # shape (1,)
                current_log_error = torch.abs(torch.log10(test_pred) - torch.log10(test_Nf.squeeze(-1)))
                # 记录（注意一致性都用 scalar）
                loss_history['log_error'].append(current_log_error.item())

            # 记录训练历史
            loss_history['epochs'].append(epoch)
            loss_history['total_loss'].append(epoch_loss)
            loss_history['data_loss'].append(epoch_data_loss)
            loss_history['physical_loss'].append(epoch_physical_loss)

            # 早停
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            if early_stop_counter >= patience and epoch >= min_epochs:
                print(f"早停: {epoch+1} 轮后没有改善")
                break

            if (epoch + 1) % 10 == 0:
                # 可视化当前预测结果（仅本折最后一个 test 样本）
                try:
                    true_val = float(test_Nf.squeeze().item())
                    pred_val = float(test_pred.squeeze().item())
                    err = abs(np.log10(pred_val) - np.log10(true_val))
                    x_vals = np.array([true_val * 0.8, true_val * 1.2])
                    lower = x_vals * (10 ** -err)
                    upper = x_vals * (10 ** err)

                    plt.figure(figsize=(6, 5))
                    plt.scatter([true_val], [pred_val], color='red', label='预测值')
                    plt.plot(x_vals, x_vals, 'k--', label='完美预测')
                    plt.fill_between(x_vals, lower, upper, alpha=0.2, label='误差带')
                    plt.xlabel('实际疲劳寿命')
                    plt.ylabel('预测疲劳寿命')
                    plt.title(f'{material_name} - 折 {test_idx+1} 第{epoch+1}轮预测')
                    plt.legend()
                    plt.grid(True)
                    plot_path = get_output_path(material_name, f"loo_fold{test_idx+1}_epoch_{epoch+1}.png")
                    plt.savefig(plot_path)
                    plt.close()
                except Exception as e:
                    print(f"可视化当前折 (样本 {test_idx+1}) 时出错: {e}")

        # 评估当前折最终结果并保存
        with torch.no_grad():
            transformer_model_fold.eval()
            pinn_model_fold.eval()
            test_features = transformer_model_fold(test_strain)
            test_combined = torch.cat([
                test_features,
                test_epsilon_a,
                test_gamma_a,
                test_FP
            ], dim=1)
            final_pred, _ = pinn_model_fold(test_combined)  # shape (1,)
            pred_scalar = float(final_pred.squeeze().item())
            true_scalar = float(test_Nf.squeeze().item())

            all_pred_lifes.append(pred_scalar)
            all_true_lifes.append(true_scalar)
            all_strain_names.append(strain_series_names[test_idx])
            all_epsilon_a_values.append(float(test_epsilon_a.squeeze().item()))
            all_gamma_a_values.append(float(test_gamma_a.squeeze().item()))
            all_fp_values.append(float(test_FP.squeeze().item()))

            log_error = abs(np.log10(pred_scalar) - np.log10(true_scalar))
            print(f"样本 {test_idx+1}: 真实寿命={true_scalar:.1f}, 预测寿命={pred_scalar:.1f}, 对数误差={log_error:.4f}")

    # 保存最终模型（使用最后一折的模型结构/参数可以考虑，也可以省略）
    final_model_path = get_output_path(material_name, f"{material_name}_final_model.pth")
    torch.save({
        'transformer_model': transformer_model_fold.state_dict(),
        'pinn_model': pinn_model_fold.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss_history': loss_history,
        'best_loss': best_loss
    }, final_model_path)

    # 保存最终预测结果到CSV文件
    results_df = pd.DataFrame({
        'strain_series_name': all_strain_names,
        'epsilon_a': all_epsilon_a_values,
        'gamma_a': all_gamma_a_values,
        'Nf': all_true_lifes,
        'FP': all_fp_values,
        'prediction': all_pred_lifes,
        'material_name': [material_name] * len(all_true_lifes)
    })
    results_path = get_output_path(material_name, f"{material_name}_final_results.csv")
    results_df.to_csv(results_path, index=False)

    # 绘制最终折的总体预测（所有样本）
    plt.figure(figsize=(10, 6))
    plt.scatter(all_true_lifes, all_true_lifes, color='blue', label='实际值')
    plt.scatter(all_true_lifes, all_pred_lifes, color='red', label='预测值')

    # 误差带（每个样本分别）
    error = np.abs(np.log10(np.array(all_pred_lifes)) - np.log10(np.array(all_true_lifes)))
    lower = np.array(all_true_lifes) * (10 ** -error)
    upper = np.array(all_true_lifes) * (10 ** error)
    plt.fill_between(
        np.array(all_true_lifes),
        lower,
        upper,
        alpha=0.2, color='gray', label='误差带'
    )

    plt.xlabel('实际疲劳寿命')
    plt.ylabel('预测疲劳寿命')
    plt.title(f'{material_name} - 最终预测结果')
    plt.legend()
    plt.grid(True)

    final_plot_path = get_output_path(material_name, f"{material_name}_final_prediction.png")
    plt.savefig(final_plot_path)
    plt.close()

    return all_pred_lifes, all_true_lifes



def train_final_model(strain_series, epsilon_a, gamma_a, Nf, FP, material_name):
    """使用所有数据训练最终模型"""
    # 移动数据到指定设备
    strain_series = strain_series.to(device)
    epsilon_a = epsilon_a.to(device)
    gamma_a = gamma_a.to(device)
    Nf = Nf.to(device)
    FP = FP.to(device)

    # 初始化模型
    lstm_input_size = 2
    lstm_hidden_size = 32
    lstm_num_layers = 2
    lstm_model = TransformerFeatureExtractor(lstm_input_size, d_model=lstm_hidden_size, nhead=2, num_layers=lstm_num_layers, dropout=0.2).to(device)

    pinn_input_size = 4
    hidden_layers = [64, 128, 64, 32]
    pinn_model = MFLP_PINN(pinn_input_size, hidden_layers, dropout=0.2).to(device)
    
    # 优化器 - 在最终模型中使用较小的学习率
    optimizer = optim.AdamW(
        list(lstm_model.parameters()) + list(pinn_model.parameters()), 
        lr=0.0005, 
        weight_decay=1e-4
    )
    
    # 训练参数
    num_epochs = 1000
    lambda_data = 1.0
    lambda_physical = 0.1
    
    # 记录损失
    loss_history = {
        "total_loss": [],
        "data_loss": [],
        "physical_loss": []
    }
    
    # 使用所有样本构造数据集
    n_samples = len(Nf)
    batch_size = min(4, n_samples)
    
    # 训练循环
    best_loss = float('inf')
    best_lstm_state = None
    best_pinn_state = None
    best_epoch = 0
    patience = 100
    early_stop_counter = 0
    
    print(f"\n开始训练最终模型 (共 {num_epochs} 轮)...")
    for epoch in tqdm(range(num_epochs)):
        # 手动构造批次
        permutation = torch.randperm(n_samples)
        epoch_loss = 0.0
        epoch_data_loss = 0.0
        epoch_physical_loss = 0.0
        
        for i in range(0, n_samples, batch_size):
            indices = permutation[i:i+batch_size]
            
            X_seq_batch = strain_series[indices].to(device)
            epsilon_a_batch = epsilon_a[indices].to(device)
            gamma_a_batch = gamma_a[indices].to(device)
            Nf_batch = Nf[indices].to(device)
            FP_batch = FP[indices].to(device)

            optimizer.zero_grad()

            FP_LSTM = lstm_model(X_seq_batch)
            X_combined = torch.cat([FP_LSTM, epsilon_a_batch, gamma_a_batch, FP_batch], dim=1)
            Np_pred, _ = pinn_model(X_combined)
            
            loss_data = rmse_loss(Nf_batch, Np_pred)
            loss_physical = fs_loss(Np_pred, FP_batch, material_name)
            loss = lambda_data * loss_data + lambda_physical * loss_physical
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lstm_model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(pinn_model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_data_loss += loss_data.item()
            epoch_physical_loss += loss_physical.item()
        
        avg_loss = epoch_loss / ((n_samples - 1) // batch_size + 1)
        avg_data_loss = epoch_data_loss / ((n_samples - 1) // batch_size + 1)
        avg_physical_loss = epoch_physical_loss / ((n_samples - 1) // batch_size + 1)
        
        loss_history["total_loss"].append(avg_loss)
        loss_history["data_loss"].append(avg_data_loss)
        loss_history["physical_loss"].append(avg_physical_loss)
        
        # 记录最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch
            best_lstm_state = lstm_model.state_dict().copy()
            best_pinn_state = pinn_model.state_dict().copy()
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        
        if early_stop_counter >= patience:
            print(f"早停: {epoch+1}轮后没有改善")
            break
        
        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Data Loss: {avg_data_loss:.4f}, Physical Loss: {avg_physical_loss:.4f}')
            
            # 每100轮保存一次中间模型
            checkpoint_path = get_output_path(material_name, f'{material_name}_model_checkpoint.pth')
            torch.save({
                'epoch': epoch,
                'lstm_state_dict': lstm_model.state_dict(),
                'pinn_state_dict': pinn_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss_history': loss_history}, 
                checkpoint_path)
    
    # 恢复到最佳模型状态
    print(f'训练完成。加载最佳模型 (Epoch {best_epoch+1}, Loss: {best_loss:.4f})')
    if best_lstm_state is not None:
        lstm_model.load_state_dict(best_lstm_state)
        pinn_model.load_state_dict(best_pinn_state)
    
    print(f'{material_name}材料模型训练完成.')
    return lstm_model, pinn_model, loss_history, None

# ----------------------------
# 结果可视化函数
# ----------------------------
def visualize_results(transformer_model, pinn_model, loss_history, dataset, material_name):
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    if "total_loss" in loss_history:
        plt.plot(loss_history["total_loss"], label="Total Loss", color='b')
        plt.plot(loss_history["data_loss"], label="Data Loss", color='r', linestyle='--')
        plt.plot(loss_history["physical_loss"], label="Physical Loss", color='g', linestyle=':')
    else:
        plt.plot(loss_history["train_loss"], label="Train Total Loss", color='b')
        plt.plot(loss_history["val_loss"], label="Val Total Loss", color='r')
        plt.plot(loss_history["train_data_loss"], label="Train Data Loss", color='b', linestyle='--')
        plt.plot(loss_history["val_data_loss"], label="Val Data Loss", color='r', linestyle='--')
        plt.plot(loss_history["train_physical_loss"], label="Train Physical Loss", color='g', linestyle=':')
        plt.plot(loss_history["val_physical_loss"], label="Val Physical Loss", color='y', linestyle=':')
        if "train_mech_loss" in loss_history:
            plt.plot(loss_history["train_mech_loss"], label="Train Mech Loss", color='c', linestyle='--')
        if "val_mech_loss" in loss_history:
            plt.plot(loss_history["val_mech_loss"], label="Val Mech Loss", color='m', linestyle='--')
    plt.xlabel("Epochs")
    plt.ylabel("Loss Value")
    plt.title(f"{material_name} Training Loss History")
    plt.legend()
    plt.grid(True)
    
    # 保存图表到输出目录
    loss_plot_path = get_output_path(material_name, f"{material_name}_loss_history.png")
    plt.savefig(loss_plot_path)
    print(f"{material_name} 损失曲线已保存到: {loss_plot_path}")
    # plt.show()  # 注释掉

    # 预测结果评估
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    with torch.no_grad():
        for batch in dataloader:
            X_seq, epsilon_a, gamma_a, Nf, FP, mech_label = batch
            # 移动数据到指定设备
            X_seq = X_seq.to(device)
            epsilon_a = epsilon_a.to(device)
            gamma_a = gamma_a.to(device)
            Nf = Nf.to(device)
            FP = FP.to(device)
            mech_label = mech_label.to(device)

            transformer_features = transformer_model(X_seq)
            X_combined = torch.cat([transformer_features, epsilon_a, gamma_a, FP], dim=1)
            Np_pred, pred_mech = pinn_model(X_combined)
            Np_pred = Np_pred.cpu().numpy()
            Nf = Nf.cpu().numpy()
            # 新增：打印测试集寿命分布
            print(f"测试集寿命分布: {Nf.flatten()}")

    # 计算评估指标
    log_error = np.abs(np.log10(Np_pred) - np.log10(Nf))
    mean_log_error = np.mean(log_error)
    max_log_error = np.max(log_error)
    pct2 = np.mean(log_error <= np.log10(2)) * 100  # 统计仍用 2× 阈值

    print(f"\n{material_name} 评估结果:")
    print(f"平均对数误差: {mean_log_error:.4f}")
    print(f"最大对数误差: {max_log_error:.4f}")
    print(f"在1.5倍误差带内的预测百分比: {pct2:.2f}%")

    # 保存预测结果
    # 1. detach / cpu / numpy
    eps_np = dataset.epsilon_a.squeeze(-1).cpu().numpy()
    gam_np = dataset.gamma_a.squeeze(-1).cpu().numpy()
    nf_np = dataset.Nf.squeeze(-1).cpu().numpy()
    pred_np = Np_pred.flatten()
    # 重新算一次对数误差
    log_err_np = np.abs(np.log10(pred_np) - np.log10(nf_np))

    # 2. 确保它们长度一致
    assert eps_np.shape[0] == gam_np.shape[0] == nf_np.shape[0] == pred_np.shape[0] == log_err_np.shape[0]

    # 3. 构造 DataFrame
    df_results = pd.DataFrame({
        "正应变幅值": eps_np,
        "剪应变幅值": gam_np,
        "实际疲劳寿命(Nf)": nf_np,
        "预测疲劳寿命(Np)": pred_np,
        "对数误差": log_err_np
    })
    # —— 替换结束 ——

    results_path = get_output_path(material_name, f"{material_name}_fatigue_life_results.csv")
    df_results.to_csv(results_path, index=False)

    # 绘制预测vs实际寿命的散点图
    plt.figure(figsize=(8, 8))
    plt.scatter(Nf, Np_pred, alpha=0.7, s=60)

    min_val = min(np.min(Nf), np.min(Np_pred)) * 0.8
    max_val = max(np.max(Nf), np.max(Np_pred)) * 1.2
    
    plt.plot([min_val, max_val], [min_val, max_val], 'r-', linewidth=2, label="完美预测")

    plt.plot([min_val, max_val], [min_val/2, max_val/2], 'k--', linewidth=1, label="1.5倍误差带")
    plt.plot([min_val, max_val], [min_val*2, max_val*2], 'k--', linewidth=1)
    
    plt.xlabel("实际疲劳寿命 (Nf)", fontsize=12)
    plt.ylabel("预测疲劳寿命 (Np)", fontsize=12)
    plt.title(f"{material_name} 疲劳寿命预测结果", fontsize=14)
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    
    # 保存图表到输出目录
    pred_plot_path = get_output_path(material_name, f"{material_name}_prediction_results.png")
    plt.savefig(pred_plot_path, dpi=300)
    print(f"{material_name} 预测结果散点图已保存到: {pred_plot_path}")
    # plt.show()  # 注释掉
    
    # 可视化Transformer特征重要性
    plt.figure(figsize=(10, 6))
    with torch.no_grad():
        transformer_model.eval()
        # 提取所有样本的特征
        transformer_features = transformer_model(dataset.strain_series.to(device))
        # 计算特征的平均重要性
        feature_importance = transformer_features.mean(dim=0).cpu().numpy()
        # 绘制特征重要性
        plt.bar(range(len(feature_importance)), feature_importance, alpha=0.7)
        plt.xlabel('特征索引', fontsize=12)
        plt.ylabel('特征重要性', fontsize=12)
        plt.title(f'{material_name} Transformer特征重要性分析', fontsize=14)
        plt.grid(True, axis='y', alpha=0.3)
    
    # 保存特征重要性图表
    feature_plot_path = get_output_path(material_name, f"{material_name}_feature_importance.png")
    plt.savefig(feature_plot_path, dpi=300)
    print(f"{material_name} 特征重要性图已保存到: {feature_plot_path}")
    # plt.show()  # 注释掉

def visualize_file_results(results, material_name):
    """可视化所有文件的预测结果"""
    if not results:
        print("没有结果可供可视化")
        return
    
    # 提取数据
    files = list(results.keys())
    epsilon_a = [results[f]['epsilon_a'] for f in files]
    gamma_a = [results[f]['gamma_a'] for f in files]
    true_life = [results[f]['Nf'] for f in files]
    pred_life = [results[f]['prediction'] for f in files]
    log_errors = [results[f]['log_error'] for f in files]
    sheet_names = [results[f]['sheet_name'] for f in files]
    
    # 按表格分组
    unique_sheets = list(set(sheet_names))
    sheet_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_sheets)))
    
    # 1. 预测vs实际寿命散点图
    plt.figure(figsize=(10, 8))
    
    for i, sheet in enumerate(unique_sheets):
        sheet_indices = [j for j, s in enumerate(sheet_names) if s == sheet]
        
        plt.scatter(
            [true_life[j] for j in sheet_indices],
            [pred_life[j] for j in sheet_indices],
            label=f"表格 {sheet}",
            color=sheet_colors[i],
            alpha=0.7,
            s=60
        )
    
    # 添加误差带
    min_val = min(min(true_life), min(pred_life)) * 0.8
    max_val = max(max(true_life), max(pred_life)) * 1.2
    
    plt.plot([min_val, max_val], [min_val, max_val], 'r-', linewidth=2, label="完美预测")
    plt.plot([min_val, max_val], [min_val/2, max_val/2], 'k--', linewidth=1, label="1.5倍误差带")
    plt.plot([min_val, max_val], [min_val*2, max_val*2], 'k--', linewidth=1)
    
    plt.xlabel("实际疲劳寿命 (Nf)", fontsize=12)
    plt.ylabel("预测疲劳寿命 (Np)", fontsize=12)
    plt.title(f"{material_name} 所有文件的疲劳寿命预测结果", fontsize=14)
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    
    pred_plot_path = get_output_path(material_name, f"{material_name}_all_files_prediction.png")
    plt.savefig(pred_plot_path, dpi=300)
    plt.show()
    
    # 2. 误差分布图
    plt.figure(figsize=(12, 6))
    
    sorted_indices = np.argsort(log_errors)
    sorted_files = [files[i] for i in sorted_indices]
    sorted_errors = [log_errors[i] for i in sorted_indices]
    sorted_sheets = [sheet_names[i] for i in sorted_indices]
    
    bar_colors = [sheet_colors[unique_sheets.index(sheet)] for sheet in sorted_sheets]
    
    plt.bar(range(len(sorted_files)), sorted_errors, color=bar_colors)
    plt.axhline(y=np.log10(2), color='r', linestyle='--', label="1.5倍误差带界限")
    
    plt.xlabel("文件索引", fontsize=12)
    plt.ylabel("对数误差", fontsize=12)
    plt.title(f"{material_name} 所有文件的预测误差排序", fontsize=14)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    error_plot_path = get_output_path(material_name, f"{material_name}_all_files_errors.png")
    plt.savefig(error_plot_path, dpi=300)
    plt.show()
    
    # 3. 误差与应变关系图
    plt.figure(figsize=(10, 8))
    plt.scatter(epsilon_a, gamma_a, c=log_errors, cmap='viridis', s=100, alpha=0.7)
    plt.colorbar(label="对数误差")
    
    plt.xlabel("正应变幅值 (ε_a)", fontsize=12)
    plt.ylabel("剪应变幅值 (γ_a)", fontsize=12)
    plt.title(f"{material_name} 预测误差与应变关系", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    strain_plot_path = get_output_path(material_name, f"{material_name}_strain_vs_error.png")
    plt.savefig(strain_plot_path, dpi=300)
    plt.show()
    
    # 4. 统计分析
    in_factor_two = sum(1 for e in log_errors if e <= np.log10(2))
    percent_in_factor_two = in_factor_two / len(log_errors) * 100
    
    print(f"\n{material_name} 预测结果统计:")
    print(f"总文件数: {len(files)}")
    print(f"平均对数误差: {np.mean(log_errors):.4f}")
    print(f"最大对数误差: {max(log_errors):.4f}")
    print(f"最小对数误差: {min(log_errors):.4f}")
    print(f"误差标准差: {np.std(log_errors):.4f}")
    print(f"在1.5倍误差带内的预测: {in_factor_two}/{len(log_errors)} ({percent_in_factor_two:.2f}%)")

# def train_models_by_individual_file(material_name='AISI316L'):
#     # 获取所有数据文件
#     data_dir = get_strain_series_path(material_name)
#     output_dir = get_output_path(material_name, '')
#     model_dir = os.path.join(output_dir, 'models')
#     csv_dir = os.path.join(output_dir, 'csv')
#     os.makedirs(model_dir, exist_ok=True)
#     os.makedirs(csv_dir, exist_ok=True)
#
#     data_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
#     if not data_files:
#         print(f"在 {data_dir} 中没有找到CSV文件")
#         return
#
#     results = []
#
#     for file_name in tqdm(data_files, desc=f"训练 {material_name} 的文件"):
#         try:
#             data_path = os.path.join(data_dir, file_name)
#             dataset = FatigueDataset(material_name=material_name)
#
#             # 创建数据加载器
#             dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
#
#             # 创建模型
#             lstm_model = FatigueLSTM(input_size=2, hidden_size=64, num_layers=2, dropout=0.2).to(device)
#             pinn_model = FatiguePINN(input_size=2, hidden_size=64, num_layers=2, dropout=0.2).to(device)
#
#             # 训练模型
#             train_single_file_model(lstm_model, pinn_model, dataset.X_seq, dataset.epsilon_a, dataset.gamma_a, dataset.Nf, dataset.FP, device)
#
#             # 预测
#             lstm_model.eval()
#             pinn_model.eval()
#             with torch.no_grad():
#                 Np_pred = lstm_model(dataset.X_seq.unsqueeze(0)).squeeze()
#                 Np_pinn = pinn_model(dataset.X_seq.unsqueeze(0)).squeeze()
#
#             # 计算误差
#             mse = F.mse_loss(Np_pred, dataset.Nf)
#             rmse = torch.sqrt(mse)
#             mae = F.l1_loss(Np_pred, dataset.Nf)
#
#             # 保存结果
#             result = {
#                 'file_name': file_name,
#                 'mse': mse.item(),
#                 'rmse': rmse.item(),
#                 'mae': mae.item(),
#                 'material': material_name
#             }
#             results.append(result)
#
#             # 保存模型
#             model_path = os.path.join(model_dir, f"{file_name.replace('.csv', '')}_model.pth")
#             torch.save({
#                 'lstm_state_dict': lstm_model.state_dict(),
#                 'pinn_state_dict': pinn_model.state_dict(),
#                 'mse': mse.item(),
#                 'rmse': rmse.item(),
#                 'mae': mae.item()
#             }, model_path)
#
#             # 每个文件训练完成后立即保存结果到CSV
#             df = pd.DataFrame(results)
#             csv_path = os.path.join(csv_dir, f"{material_name}_results.csv")
#             df.to_csv(csv_path, index=False)
#             print(f"已保存 {material_name} 的结果到 {csv_path}")
#
#         except Exception as e:
#             print(f"处理文件 {file_name} 时出错: {str(e)}")
#             continue
#
#     print(f"完成 {material_name} 的训练")
#     return results

def train_all_files_together(material_name, model_params=None, training_params=None):
    print(f"\n开始使用所有文件数据一起训练{material_name}材料的模型...")
    
    data_dir = get_strain_series_path(material_name)
    if not data_dir or not os.path.exists(data_dir):
        raise FileNotFoundError(f"找不到应变时间序列文件夹: {data_dir}")
    
    csv_files = [f for f in os.listdir(data_dir) 
                if f.startswith('strain_series_') and f.endswith('.csv')]
    print(f"找到 {len(csv_files)} 个CSV文件")
    
    fatigue_data_dir = get_fatigue_data_path(material_name)
    fatigue_data_path = None
    for file in os.listdir(fatigue_data_dir):
        if file.endswith('.xls') and material_name in file:
            fatigue_data_path = os.path.join(fatigue_data_dir, file)
            print(f"找到匹配的疲劳数据文件: {file}")
            break
    if not fatigue_data_path:
        raise FileNotFoundError(f"无法在 {fatigue_data_dir} 中找到包含 {material_name} 的疲劳数据文件")
    
    # 读取疲劳数据
    epsilon_a_all, gamma_a_all, Nf_all, _, _, FP_all = read_fatigue_data(fatigue_data_path)
    
    # 收集所有数据
    all_data = []
    max_sequence_length = 0
    
    # 首先确定最大序列长度
    for csv_file in csv_files:
        file_path = os.path.join(data_dir, csv_file)
        try:
            # 从文件名解析应变值
            epsilon_a, gamma_a = parse_strain_values_from_filename(csv_file)
            
            # 寻找匹配的疲劳数据
            tolerance = 1e-5
            mask = (abs(epsilon_a_all - epsilon_a) < tolerance) & (abs(gamma_a_all - gamma_a) < tolerance)
            matching_indices = np.where(mask)[0]
            
            if len(matching_indices) >= 1:
                idx = matching_indices[0]
                
                # 读取时间序列数据
                time, normal_strain, shear_strain = read_strain_series(file_path)

                # 填充或截断序列到最大长度
                if len(normal_strain) < max_sequence_length:
                    # 填充
                    normal_strain = np.pad(normal_strain, (0, max_sequence_length - len(normal_strain)), 'constant')
                    shear_strain = np.pad(shear_strain, (0, max_sequence_length - len(shear_strain)), 'constant')
                else:
                    # 截断
                    normal_strain = normal_strain[:max_sequence_length]
                    shear_strain = shear_strain[:max_sequence_length]
                
                strain_series = np.stack([normal_strain, shear_strain], axis=1)

                all_data.append({
                    'strain_series': strain_series,
                    'epsilon_a': epsilon_a,
                    'gamma_a': gamma_a,
                    'Nf': Nf_all[idx],
                    'FP': FP_all[idx]
                })
                print(f"成功加载文件 {csv_file} 的数据")
            else:
                print(f"警告: 文件 {csv_file} 没有找到匹配的疲劳数据，跳过")
                
        except Exception as e:
            print(f"处理文件 {csv_file} 时出错: {str(e)}")
            continue
                    
    if not all_data:
        raise ValueError("没有找到任何有效的训练数据！")
    
    # 更高效地转换为张量
    print("\n转换数据为PyTorch张量...")
    
    # 首先转换为numpy数组
    strain_series_np = np.array([d['strain_series'] for d in all_data])
    epsilon_a_np = np.array([d['epsilon_a'] for d in all_data]).reshape(-1, 1)
    gamma_a_np = np.array([d['gamma_a'] for d in all_data]).reshape(-1, 1)
    Nf_np = np.array([d['Nf'] for d in all_data]).reshape(-1, 1)
    FP_np = np.array([d['FP'] for d in all_data]).reshape(-1, 1)
    
    # 然后一次性转换为张量
    strain_series = torch.from_numpy(strain_series_np).float()
    epsilon_a = torch.from_numpy(epsilon_a_np).float()
    gamma_a = torch.from_numpy(gamma_a_np).float()
    Nf = torch.from_numpy(Nf_np).float()
    FP = torch.from_numpy(FP_np).float()
    
    print(f"数据形状:")
    print(f"- 应变序列: {strain_series.shape}")
    print(f"- 正应变: {epsilon_a.shape}")
    print(f"- 剪应变: {gamma_a.shape}")
    print(f"- 疲劳寿命: {Nf.shape}")
    print(f"- 疲劳参数: {FP.shape}")
    
    # 创建数据集和数据加载器
    dataset = torch.utils.data.TensorDataset(strain_series, epsilon_a, gamma_a, Nf, FP)
    
    # 设置默认模型参数
    if model_params is None:
        model_params = {
            'transformer': {
                'input_size': 2,
                'd_model': 64,
                'nhead': 4,
                'num_layers': 2,
                'dropout': 0.1
            },
            'pinn': {
                'input_size': 4,
                'hidden_layers': [128, 256, 128, 64],
                'dropout': 0.2
            }
        }
                
                # 初始化模型
    transformer_model = TransformerFeatureExtractor(
        input_size=model_params['transformer']['input_size'],
        d_model=model_params['transformer']['d_model'],
        nhead=model_params['transformer']['nhead'],
        num_layers=model_params['transformer']['num_layers'],
        dropout=model_params['transformer']['dropout']
    ).to(device)

    pinn_model = MFLP_PINN(
        input_size=model_params['pinn']['input_size'],
        hidden_layers=model_params['pinn']['hidden_layers'],
        dropout=model_params['pinn']['dropout']
    ).to(device)
    # 创建数据加载器
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=training_params['batch_size'],
        shuffle=True
    )
    
    # 初始化优化器
    optimizer = optim.AdamW(
        list(transformer_model.parameters()) + list(pinn_model.parameters()),
        lr=training_params['learning_rate'],
        weight_decay=training_params['weight_decay']
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=training_params['scheduler_factor'],
        patience=training_params['scheduler_patience'],
        min_lr=training_params['min_lr']
    )
    
    # 训练循环
    loss_history = {
        'total_loss': [],
        'data_loss': [],
        'physical_loss': []
    }
    
    best_loss = float('inf')
    best_epoch = 0
    early_stop_counter = 0
    
    print("\n开始训练模型...")
    
    # 创建实时可视化图表
    plt.ion()  # 打开交互模式
    fig = plt.figure(figsize=(15, 10))
    
    # 记录每个batch的损失
    batch_losses = {
        'total_loss': [],
        'data_loss': [],
        'physical_loss': []
    }
    
    for epoch in range(training_params['num_epochs']):
        epoch_loss = 0.0
        epoch_data_loss = 0.0
        epoch_physical_loss = 0.0
        
        for batch_idx, batch in enumerate(dataloader):
            X_seq, epsilon_a_batch, gamma_a_batch, Nf_batch, FP_batch = batch

            # 移动数据到指定设备
            X_seq = X_seq.to(device)
            epsilon_a_batch = epsilon_a_batch.to(device)
            gamma_a_batch = gamma_a_batch.to(device)
            Nf_batch = Nf_batch.to(device)
            FP_batch = FP_batch.to(device)

            # 前向传播
            transformer_features = transformer_model(X_seq)
            X_combined = torch.cat([transformer_features, epsilon_a_batch, gamma_a_batch, FP_batch], dim=1)
            Np_pred, _ = pinn_model(X_combined)
            
            # 计算损失
            data_loss = rmse_loss(Nf_batch, Np_pred)
            physical_loss = fs_loss(Np_pred, FP_batch, material_name)
            loss = training_params['lambda_data'] * data_loss + training_params['lambda_physical'] * physical_loss
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 记录batch损失
            batch_losses['total_loss'].append(loss.item())
            batch_losses['data_loss'].append(data_loss.item())
            batch_losses['physical_loss'].append(physical_loss.item())
            
            epoch_loss += loss.item()
            epoch_data_loss += data_loss.item()
            epoch_physical_loss += physical_loss.item()
            
            # 实时更新可视化
            if batch_idx % 2 == 0:  # 每2个batch更新一次，避免更新太频繁
                plt.clf()  # 清除当前图表
                
                # 创建子图
                plt.subplot(2, 2, 1)
                plt.plot(batch_losses['total_loss'], 'b-', label='Total Loss', alpha=0.6)
                plt.plot(batch_losses['data_loss'], 'r--', label='Data Loss', alpha=0.6)
                plt.plot(batch_losses['physical_loss'], 'g:', label='Physical Loss', alpha=0.6)
                plt.yscale('log')
                plt.xlabel('Batch')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True)
                plt.title(f'Training Loss History (Epoch {epoch+1})')
                
                # 预测vs真实值散点图
                plt.subplot(2, 2, 2)
                with torch.no_grad():
                    transformer_model.eval()
                    pinn_model.eval()
                    all_features = transformer_model(strain_series)
                    X_combined = torch.cat([all_features, epsilon_a, gamma_a, FP], dim=1)
                    all_predictions = pinn_model(X_combined)
                    
        
        # 计算平均损失
        avg_loss = epoch_loss / len(dataloader)
        avg_data_loss = epoch_data_loss / len(dataloader)
        avg_physical_loss = epoch_physical_loss / len(dataloader)
        
        # 更新学习率
        scheduler.step(avg_loss)
        
        # 记录损失历史
        loss_history['total_loss'].append(avg_loss)
        loss_history['data_loss'].append(avg_data_loss)
        loss_history['physical_loss'].append(avg_physical_loss)
        
        # 早停检查
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch
            best_transformer_state = transformer_model.state_dict().copy()
            best_pinn_state = pinn_model.state_dict().copy()
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        
        if early_stop_counter >= training_params['patience'] and epoch >= training_params['min_epochs']:
            print(f"\n早停: {epoch+1}轮后没有改善")
            break
        
        # 更新可视化（每10轮更新一次）
        if (epoch + 1) % 10 == 0 or epoch == 0:
            plt.clf()  # 清除当前图表
            
            # 创建子图
            plt.subplot(2, 2, 1)
            plt.plot(loss_history['total_loss'], 'b-', label='Total Loss')
            plt.plot(loss_history['data_loss'], 'r--', label='Data Loss')
            plt.plot(loss_history['physical_loss'], 'g:', label='Physical Loss')
            plt.yscale('log')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            plt.title('Training Loss History')
            
            # 预测vs真实值散点图
            plt.subplot(2, 2, 2)
            with torch.no_grad():
                transformer_model.eval()
                pinn_model.eval()
                all_features = transformer_model(strain_series)
                X_combined = torch.cat([all_features, epsilon_a, gamma_a, FP], dim=1)
                all_predictions = pinn_model(X_combined)
                
                plt.scatter(Nf.cpu().numpy(), all_predictions.cpu().numpy(), c='b', label='Predictions')
                min_val = min(Nf.min().item(), all_predictions.min().item())
                max_val = max(Nf.max().item(), all_predictions.max().item())
                plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
                plt.plot([min_val, max_val], [min_val/2, max_val/2], 'k:', label='Factor of 2')
                plt.plot([min_val, max_val], [min_val*2, max_val*2], 'k:')
                plt.xscale('log')
                plt.yscale('log')
                plt.xlabel('True Life')
                plt.ylabel('Predicted Life')
                plt.legend()
                plt.grid(True)
                plt.title('Prediction vs True Life')
                
                transformer_model.train()
                pinn_model.train()
            
            # 学习率
            plt.subplot(2, 2, 3)
            current_lr = optimizer.param_groups[0]['lr']
            plt.scatter(epoch, current_lr, c='b')
            plt.yscale('log')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.grid(True)
            plt.title('Learning Rate')
            
            # 对数误差分布
            plt.subplot(2, 2, 4)
            with torch.no_grad():
                log_errors = torch.abs(torch.log10(all_predictions) - torch.log10(Nf))
                plt.hist(log_errors.cpu().numpy(), bins=20, color='b', alpha=0.7)
                plt.axvline(x=np.log10(2), color='r', linestyle='--', label='Factor of 2')
                plt.xlabel('Log Error')
                plt.ylabel('Count')
                plt.legend()
                plt.grid(True)
                plt.title('Log Error Distribution')
            
            plt.tight_layout()
            plt.draw()  # 使用draw()而不是pause()
            plt.pause(0.1)  # 短暂暂停以更新图表
            
            # 每100轮保存一次中间模型
            if (epoch + 1) % 100 == 0:
                checkpoint_path = get_output_path(material_name, f'{material_name}_model_checkpoint.pth')
                torch.save({
                    'epoch': epoch,
                    'transformer_state_dict': transformer_model.state_dict(),
                    'pinn_state_dict': pinn_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss_history': loss_history}, 
                    checkpoint_path)
    
    plt.ioff()  # 关闭交互模式
    plt.show()  # 显示最终图表
    
    print(f"\n训练完成，最佳模型在第 {best_epoch+1} 轮")
    
    # 恢复最佳模型
    if best_transformer_state is not None:
        transformer_model.load_state_dict(best_transformer_state)
        pinn_model.load_state_dict(best_pinn_state)

# ----------------------------
# 主程序
# ----------------------------
if __name__ == "__main__":
    # 设置中文字体支持
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
    except:
        print("警告: 无法设置中文字体，图表中文可能无法正确显示")
    
    # 处理所有材料
    # materials = ['Q235B2']
    materials = ['AISI316L', 'GH4169', 'TC4', 'CuZn37', 'Q235B1', 'Q235B2']
    available_materials = []
    
    # 先检查哪些材料有可用的应变时间序列数据
    print("\n检查材料数据可用性：")
    for material_name in materials:
        strain_series_path = get_strain_series_path(material_name)
        if os.path.exists(strain_series_path):
            # 检查是否有文件
            files = [f for f in os.listdir(strain_series_path) 
                    if f.startswith('strain_series_') and (f.endswith('.xls') or f.endswith('.csv'))]
            if files:
                available_materials.append(material_name)
                print(f"✓ {material_name}: 找到 {len(files)} 个应变时间序列文件")
            else:
                print(f"× {material_name}: 应变时间序列文件夹存在，但没有找到有效的数据文件")
        else:
            print(f"× {material_name}: 找不到应变时间序列数据文件夹")
    
    if not available_materials:
        print("\n错误：没有找到任何材料的应变时间序列数据！")
        exit(1)
    
    print(f"\n将处理以下材料: {', '.join(available_materials)}")
    
    # 对每个材料进行训练
    for material_name in available_materials:
        print(f"\n{'='*50}")
        print(f"开始处理 {material_name} 材料数据")
        print(f"{'='*50}")


        #             # 先训练一次常规模型
        # transformer_model, pinn_model, loss_history, dataset = train_model(material_name)
        #     # 然后对同一个 dataset 调用留一交叉验证
        # all_preds, all_trues = train_with_leave_one_out(transformer_model, pinn_model, dataset, material_name)

        #     # 可视化训练结果
        # visualize_results(transformer_model, pinn_model, loss_history, dataset, material_name)
            
        # print(f"\n{material_name} 材料处理完成！")
        # print(f"结果已保存到 results/{material_name}/ 目录下")
        # print(f"可以查看以下文件：")
        # print(f"1. {material_name}_prediction_results.csv - 详细的预测数据")
        # print(f"2. {material_name}_prediction_plot.png - 预测结果可视化")
        # print(f"3. {material_name}_loss_history.png - 训练损失曲线")

        
        try:
            # 先训练一次常规模型
            transformer_model, pinn_model, loss_history, dataset = train_model(material_name)
            # 然后对同一个 dataset 调用留一交叉验证
            all_preds, all_trues = train_with_leave_one_out(transformer_model, pinn_model, dataset, material_name)

            # 可视化训练结果
            visualize_results(transformer_model, pinn_model, loss_history, dataset, material_name)
            
            print(f"\n{material_name} 材料处理完成！")
            print(f"结果已保存到 results/{material_name}/ 目录下")
            print(f"可以查看以下文件：")
            print(f"1. {material_name}_prediction_results.csv - 详细的预测数据")
            print(f"2. {material_name}_prediction_plot.png - 预测结果可视化")
            print(f"3. {material_name}_loss_history.png - 训练损失曲线")
            
        except Exception as e:
            print(f"\n处理 {material_name} 时发生错误：")
            print(str(e))
            import traceback


