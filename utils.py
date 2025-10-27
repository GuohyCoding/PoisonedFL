"""
utils.py
========
该模块提供联邦学习实验所需的常用工具函数，如余弦相似度计算、
梯度中值聚合以及 LEAF 格式数据读取助手。
"""

import mxnet as mx
from mxnet import nd, autograd, gluon
import numpy as np
import json
from collections import defaultdict
import os


def cal_cos(grad_1, grad_2):
    """
    简要概述: 计算两个梯度向量的余弦相似度。

    参数:
        grad_1 (nd.NDArray): 第一个梯度向量。
        grad_2 (nd.NDArray): 第二个梯度向量。

    返回:
        nd.NDArray: 单元素张量，表示余弦相似度值。

    异常:
        无。

    复杂度:
        时间 O(d); 空间 O(1)。

    费曼学习法:
        (A) 功能: 衡量两个梯度方向的接近程度。
        (B) 类比: 像比较两支箭是否指向相似方向，越平行值越接近 1。
        (C) 步骤拆解:
            1. 计算两个向量的点积，度量方向一致性。
            2. 分别求出向量范数，避免长度影响结果。
            3. 将点积除以范数乘积，再加入微小平滑项防止除零。
        (D) 示例:
            - 调用: `sim = cal_cos(nd.array([1,0]), nd.array([0.5,0]))`
            - 输出: `sim` 约等于 1。
        (E) 边界与测试: 输入向量不应全为零；可测试正交向量得到相似度约 0。
        (F) 背景与参考: 来源于向量空间模型，可参考《Introduction to Information Retrieval》。 
    """
    # 使用平滑项避免零向量造成除零错误
    return nd.dot(grad_1, grad_2)/(nd.norm(grad_1) + 1e-9) / (nd.norm(grad_2) + 1e-9)


def median_grad(gradients):
    """
    简要概述: 对一组梯度按维度取中位数，提供鲁棒聚合基线。

    参数:
        gradients (list[list[nd.NDArray]]): 各客户端梯度列表，按参数块组织。

    返回:
        nd.NDArray: 扁平化后的中值梯度向量。

    异常:
        无。

    复杂度:
        时间 O(d·n log n); 空间 O(d)，其中 d 为参数维度，n 为客户端数。

    费曼学习法:
        (A) 功能: 给每个参数维度选出居中的值，减少极端梯度对结果的影响。
        (B) 类比: 像对多份测量结果排队，取中间的那个更可靠。
        (C) 步骤拆解:
            1. 将每个客户端的梯度拼接成列向量，便于逐维排序。
            2. 对所有列按维度排序，找到中间位置。
            3. 对偶数列时取中间两列平均，对奇数列直接取中间列。
        (D) 示例:
            - 调用: `median = median_grad([[nd.array([1,2])],[nd.array([2,3])]])`
            - 输出: `median` 为中位梯度向量。
        (E) 边界与测试: 保证所有梯度尺寸一致；可测试奇偶客户端数量的中位数一致性。
        (F) 背景与参考: 属于鲁棒统计方法，可参考《Robust Statistics》。
    """
    # 将每个客户端的梯度展平拼接为列向量
    param_list = [nd.concat(*[xx.reshape((-1, 1))
                            for xx in x], dim=0) for x in gradients]
    sorted_arr = nd.sort(nd.concat(*param_list, dim=1), axis=-1)
    median_idx = sorted_arr.shape[-1] // 2
    if sorted_arr.shape[-1] % 2 == 0:
        # 偶数个客户端时取中间两个值的平均
        median = mx.nd.mean(sorted_arr[:,median_idx-1:median_idx], axis=-1)
    else:
        # 奇数个客户端时取中位列
        median = mx.nd.take(sorted_arr, median_idx, axis=-1)

    return median


def read_dir(data_dir):
    """
    简要概述: 读取 LEAF 数据集目录，汇总客户端、群组与数据内容。

    参数:
        data_dir (str): 包含 JSON 文件的目录路径。

    返回:
        tuple[list[str], list[str], dict]: 客户端列表、群组列表与用户数据字典。

    异常:
        FileNotFoundError: 当目录不存在时由 `os.listdir` 抛出。

    复杂度:
        时间 O(F·S); 空间 O(S)，其中 F 为文件数，S 为 JSON 大小。

    费曼学习法:
        (A) 功能: 将目录中的 JSON 文件解析成统一的数据结构，便于后续索引。
        (B) 类比: 像整理多份学生成绩单，合并成总花名册与各自成绩表。
        (C) 步骤拆解:
            1. 列出目录内所有 JSON 文件。
            2. 逐个打开并解析，收集用户、群组和用户数据。
            3. 对客户端列表进行排序，保持确定性顺序。
        (D) 示例:
            - 调用: `clients, groups, data = read_dir('./leaf/train')`
            - 输出: `clients` 为排序后的用户 id 列表。
        (E) 边界与测试: 需确保 JSON 含有 `'users'` 和 `'user_data'`; 可构造小型目录进行单元测试。
        (F) 背景与参考: 适用于 LEAF/FEMNIST 数据格式，可参考 LEAF 官方说明。
    """
    clients = []
    groups = []
    data = defaultdict(lambda: None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        data.update(cdata['user_data'])

    clients = list(sorted(data.keys()))
    return clients, groups, data


def read_data(train_data_dir, test_data_dir):
    """
    简要概述: 读取训练与测试目录，返回对齐的客户端数据映射。

    参数:
        train_data_dir (str): 训练数据目录。
        test_data_dir (str): 测试数据目录。

    返回:
        tuple[list[str], list[str], dict, dict]: 客户端列表、群组列表、训练与测试数据映射。

    异常:
        AssertionError: 当训练与测试客户端或群组不一致时触发。

    复杂度:
        时间 O(F·S); 空间 O(S)，其中 F、S 同上。

    费曼学习法:
        (A) 功能: 将训练和测试的 JSON 数据读取并确保客户端集合一致。
        (B) 类比: 像分别统计上半年与下半年成绩后确认学生名单一致，再汇总比较。
        (C) 步骤拆解:
            1. 调用 `read_dir` 分别读取训练与测试目录。
            2. 断言两侧的客户端与群组列表一致，保证配对正确。
            3. 返回统一的客户端、群组以及训练、测试数据字典。
        (D) 示例:
            - 调用: `clients, groups, train_data, test_data = read_data(train_dir, test_dir)`
            - 输出: 四个对象分别对应客户端、群组和数据映射。
        (E) 边界与测试: 若存在缺失客户端会触发断言；可为不一致情况编写负向测试。
        (F) 背景与参考: 用于 FEMNIST 等基于 LEAF 的数据集。
    """
    train_clients, train_groups, train_data = read_dir(train_data_dir)
    test_clients, test_groups, test_data = read_dir(test_data_dir)

    assert train_clients == test_clients
    assert train_groups == test_groups

    return train_clients, train_groups, train_data, test_data


__AI_ANNOTATION_SUMMARY__ = """
cal_cos: 计算两个梯度向量之间的余弦相似度。
median_grad: 对多客户端梯度取逐维中位数实现鲁棒聚合。
read_dir: 解析 LEAF 目录并汇总客户端与数据映射。
read_data: 从训练/测试目录读取并校验客户端一致性。
"""
