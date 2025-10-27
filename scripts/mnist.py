"""
mnist.py
========
脚本批量发起 MNIST 数据集上的 PoisonedFL 攻击实验，覆盖三种聚合器配置。
"""

import os

# 中值聚合基线
os.system("python test_agr.py --dataset mnist --gpu 7 --net cnn --niter 6000 --nworkers 1200 --nfake 240 --aggregation median --byz_type poisonedfl --sf 8 --local_epoch 1 ")

# 截断均值基线
os.system("python test_agr.py --dataset mnist --gpu 8 --net cnn --niter 6000 --nworkers 1200 --nfake 240 --aggregation trim --byz_type poisonedfl --sf 8 --local_epoch 1 ")

# 范数归一化均值基线
os.system("python test_agr.py --dataset mnist --gpu 3 --net cnn --niter 6000 --nworkers 1200 --nfake 240 --aggregation mean_norm --byz_type poisonedfl --sf 8 --local_epoch 1 ")


__AI_ANNOTATION_SUMMARY__ = """
<无函数>: 脚本包含 3 条 MNIST 实验命令，用于比较聚合策略。
"""
