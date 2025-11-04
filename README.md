# PoisonedFL (PyTorch)

联邦学习项目骨架，基于 Python 3.10+ 与 PyTorch 2.x，支持模块化组件（数据集、模型、训练、攻击、防御、聚合、工具、配置、测试）。

## 目录结构

```
federated/
  __init__.py
  aggregators/
  attacks/
  configs/
  datasets/
  defenses/
  models/
  training/
  utils/
minimal_run.py
tests/
```

## 环境准备

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## 最小示例

```bash
python minimal_run.py --dataset mnist --model simple_cnn --trainer fedavg --rounds 1 --num-clients 2
```

### 可选参数

- `--attack random_noise`：启用随机噪声攻击。
- `--defense median_clip`：启用中值截断防御。
- `--device cuda:0`：指定训练设备。

## 测试

```bash
pytest tests/test_fedavg.py
```
