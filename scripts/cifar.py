"""
cifar.py
========
脚本用于批量启动针对 CIFAR-10 数据集的不同聚合策略实验，便于对比 PoisonedFL 攻击效果。
"""

import os

# 运行 PoisonedFL + 中值聚合实验
os.system("python test_agr.py --dataset cifar10 --gpu 6 --net cnn_cifar --niter 10000 --nworkers 1200 --nfake 240 --aggregation median --byz_type poisonedfl --sf 8 --local_epoch 1 --bias 0.2  --lr 0.03")

# 运行 PoisonedFL + 截断均值实验
os.system("python test_agr.py --dataset cifar10 --gpu 4 --net cnn_cifar --niter 10000 --nworkers 1200 --nfake 240 --aggregation trim --byz_type poisonedfl --sf 8 --local_epoch 1 --bias 0.2  --lr 0.03")

# 运行 PoisonedFL + 范数归一化均值实验
os.system("python test_agr.py --dataset cifar10 --gpu 5 --net cnn_cifar --niter 10000 --nworkers 1200 --nfake 240 --aggregation mean_norm --byz_type poisonedfl --sf 8 --local_epoch 1 --bias 0.2  --lr 0.03")


__AI_ANNOTATION_SUMMARY__ = """
<无函数>: 脚本包含 3 条 CIFAR-10 实验命令，分别比较不同聚合策略。
"""

