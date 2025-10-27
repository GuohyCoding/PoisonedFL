"""
purchase.py
===========
脚本发起 Purchase 数据集上的 PoisonedFL 攻击实验，比较三种聚合方案。
"""

import os

# 中值聚合实验
os.system("python test_agr.py --dataset purchase --gpu 0 --net dnn --niter 6000 --nworkers 1200 --nfake 240 --aggregation median --byz_type poisonedfl --sf 8 --local_epoch 1 --batch_size 128 --lr 0.1")

# 截断均值实验
os.system("python test_agr.py --dataset purchase --gpu 1 --net dnn --niter 6000 --nworkers 1200 --nfake 240 --aggregation trim --byz_type poisonedfl --sf 8 --local_epoch 1 --batch_size 128 --lr 0.1")

# 范数归一化均值实验
os.system("python test_agr.py --dataset purchase --gpu 2 --net dnn --niter 6000 --nworkers 1200 --nfake 240 --aggregation mean_norm --byz_type poisonedfl --sf 8 --local_epoch 1 --batch_size 128 --lr 0.1")


__AI_ANNOTATION_SUMMARY__ = """
<无函数>: 脚本包含 3 条 Purchase 数据集实验命令。
"""
