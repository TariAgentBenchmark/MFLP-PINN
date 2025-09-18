## 环境搭建

损失函数和物理定义加上了，有个小请求：
如果感觉我做的好的话。

平台有抽成，可以说我做的不满意，申请退款，把钱私发给我，这样我可以多赚个钱。
如果可以可以加我v聊一下：15637899910。
如果不行也麻烦不要在群里说，我们还正常走平台也可以，材料参数和物理损失我都加上了。

- 使用 uv 管理 Python 解释器与依赖（未安装 uv 可执行）：
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

- 在项目根目录同步环境（将按 pyproject.toml 自动安装 Python 3.12 和依赖）：
```bash
cd /home/hanbing/data/parttime/MFLP-PINN
uv sync
```

- 或使用 pip 安装依赖（阿里云镜像）：
```bash
pip install -i https://mirrors.aliyun.com/pypi/simple/ -r req.txt
```

## 运行程序

### Bayesian 时间序列（不确定性估计）

- ALL 材料，随机划分 train/test：
```bash
uv run python /home/hanbing/data/parttime/MFLP-PINN/bayesian_timeseries.py --material ALL --method split --test-ratio 0.2 --seed 42
```

- 单一材料，留一验证（LOO）：
```bash
uv run python /home/hanbing/data/parttime/MFLP-PINN/bayesian_timeseries.py --material AISI316L --method loo --ci 0.95
```

- 输出位置：`results/<材料名>/<材料名>_bayes_timeseries_*`（csv/png）

### MFLP-PINN（物理约束 PINN）

- ALL 材料，train/test：
```bash
uv run python /home/hanbing/data/parttime/MFLP-PINN/mflp-pinn.py --material ALL --method split --hidden-dims 128,64 --epochs 1500 --upper-cycles 1e7
```

- 单一材料，LOO（脚本内默认缩短轮数以加速）：
```bash
uv run python /home/hanbing/data/parttime/MFLP-PINN/mflp-pinn.py --material AISI316L --method loo
```

- 可选参数：`--no-fp` 不使用 FP 特征；`--device cuda` 强制用 GPU。

- 输出位置：`results/<材料名>/<材料名>_mflp_pinn_*`（csv/png）
