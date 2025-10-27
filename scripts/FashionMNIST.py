"""
FashionMNIST.py
===============
脚本批量启动针对 FashionMNIST 数据集的 PoisonedFL 攻击实验，比较不同聚合策略表现。
"""

import os

# 中值聚合对比实验
os.system("python test_agr.py --dataset FashionMNIST --gpu 0 --net cnn --niter 6000 --nworkers 1200 --nfake 240 --aggregation median --byz_type poisonedfl --sf 8 --local_epoch 1 ")

# 截断均值对比实验
os.system("python test_agr.py --dataset FashionMNIST --gpu 0 --net cnn --niter 6000 --nworkers 1200 --nfake 240 --aggregation trim --byz_type poisonedfl --sf 8 --local_epoch 1 ")

# 范数归一化均值对比实验
os.system("python test_agr.py --dataset FashionMNIST --gpu 0 --net cnn --niter 6000 --nworkers 1200 --nfake 240 --aggregation mean_norm --byz_type poisonedfl --sf 8 --local_epoch 1 ")


__AI_ANNOTATION_SUMMARY__ = """
<无函数>: 脚本包含 3 条 FashionMNIST 实验命令，分别对应三种聚合策略。
"""
