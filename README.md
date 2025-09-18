## 环境搭建

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

- 启动训练：
```bash
uv run python main.py
```

- 结果将保存到 `results/<材料名>/`。
