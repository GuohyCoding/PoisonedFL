"""
FEMNIST.py
==========
脚本用于运行 FEMNIST 数据集上的 PoisonedFL 攻击实验，覆盖常见鲁棒聚合策略。
"""

import os

# 中值聚合实验
os.system("python test_agr.py --dataset FEMNIST --gpu 0 --net cnn --niter 10000 --nworkers 1200 --nfake 240 --aggregation median --byz_type poisonedfl --sf 8 --local_epoch 1")

# 截断均值实验
os.system("python test_agr.py --dataset FEMNIST --gpu 0 --net cnn --niter 10000 --nworkers 1200 --nfake 240 --aggregation trim --byz_type poisonedfl --sf 8 --local_epoch 1")

# 范数归一化均值实验
os.system("python test_agr.py --dataset FEMNIST --gpu 0 --net cnn --niter 10000 --nworkers 1200 --nfake 240 --aggregation mean_norm --byz_type poisonedfl --sf 8 --local_epoch 1")


__AI_ANNOTATION_SUMMARY__ = """
<无函数>: 脚本包含 3 条 FEMNIST 攻击实验命令，用于比较聚合策略。
"""
