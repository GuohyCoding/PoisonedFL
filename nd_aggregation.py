"""
nd_aggregation.py
=================
提供联邦学习中的多种聚合函数以及若干辅助计算工具，便于在存在
拜占庭攻击的假设下评估和更新全局模型。
"""

import mxnet as mx
from mxnet import nd, autograd, gluon
import numpy as np
import byzantine
import wandb
from sklearn.metrics import roc_auc_score
# import hdbscan


def block_wise_median(param_values):
    """
    简要概述: 对参数块按列取中位数，用于大规模梯度的分块稳健聚合。

    参数:
        param_values (nd.NDArray): 形状为 (块大小, 客户端数) 的二维矩阵。

    返回:
        nd.NDArray: 每行的中位数，形状为 (块大小,)。

    异常:
        无。

    复杂度:
        时间 O(k log m); 空间 O(1)，其中 k 为块大小，m 为客户端数。

    费曼学习法:
        (A) 功能: 给每个参数维度挑选“居中的”值，抵御极端梯度。
        (B) 类比: 像统计每道题的多份答案，将它们排序后取最中间的那份。
        (C) 步骤拆解:
            1. 对每行数据排序，将异常值推到两端。
            2. 取排序后中间位置的元素作为中位数。
        (D) 示例:
            - 调用: `block_wise_median(nd.array([[1, 3, 2], [4, 6, 5]]))`
            - 输出: `nd.array([2., 5.])`。
        (E) 边界与测试: 需保证客户端数量大于 0；可测试奇偶客户端数量的差异。
        (F) 背景与参考: 属鲁棒统计方法，可参考《Robust Statistics》。
    """
    return param_values.sort(axis=-1)[:, param_values.shape[-1] // 2]


def block_wise_trim(param_values, b, m):
    """
    简要概述: 对参数块执行截断平均，剔除两侧 b 列后对剩余 m 列求均值。

    参数:
        param_values (nd.NDArray): 形状为 (块大小, 客户端数) 的矩阵。
        b (int): 需剔除的左右两端列数。
        m (int): 保留列的数量，等于总列数减去 2b。

    返回:
        nd.NDArray: 截断后的均值，形状为 (块大小,)。

    异常:
        无。

    复杂度:
        时间 O(k log m); 空间 O(1)。

    费曼学习法:
        (A) 功能: 丢掉最极端的值再平均，降低恶意梯度影响。
        (B) 类比: 裁判打分去掉最高和最低分，避免结果被极端分扭曲。
        (C) 步骤拆解:
            1. 对每行数据排序，让异常值落在两端。
            2. 选择中间连续的 m 列。
            3. 对保留部分求平均得到代表值。
        (D) 示例:
            - 调用: `block_wise_trim(nd.array([[1, 2, 100, 3]]), 1, 2)`
            - 输出: `nd.array([2.5])`。
        (E) 边界与测试: 要求 `m > 0`；可测试 `b=0` 时退化为均值。
        (F) 背景与参考: 与 Trimmed Mean 聚合方法一致，可参考《Robust Estimation》。
    """
    return param_values.sort(axis=-1)[:, b:(b+m)].mean(axis=-1)


def cos_sim_nd(p, q):
    """
    简要概述: 计算两个向量的余弦距离 (1 - 余弦相似度)。

    参数:
        p (nd.NDArray): 向量一。
        q (nd.NDArray): 向量二。

    返回:
        nd.NDArray: 单值张量，表示余弦距离。

    异常:
        无。

    复杂度:
        时间 O(d); 空间 O(1)。

    费曼学习法:
        (A) 功能: 判断两个向量方向是否一致。
        (B) 类比: 比较两支箭头的指向，越平行距离越小。
        (C) 步骤拆解:
            1. 计算向量点积衡量方向一致度。
            2. 计算向量范数并进行归一化。
            3. 用 1 减去相似度得到距离。
        (D) 示例:
            - 调用: `cos_sim_nd(nd.array([1,0]), nd.array([0.5,0]))`
            - 输出: 接近 0。
        (E) 边界与测试: 输入不应为零向量；可测试正交向量得到距离约 1。
        (F) 背景与参考: 参见《Introduction to Information Retrieval》关于余弦距离的说明。
    """
    return 1 - (p * q / (p.norm() * q.norm())).sum()


def median(gradients, net, lr, nfake, byz, history, fixed_rand, init_model, last_50_model, last_grad, sf, e):
    """
    简要概述: 执行中值聚合并允许拜占庭攻击函数预先篡改恶意梯度。

    参数:
        gradients (list[list[nd.NDArray]]): 客户端梯度列表。
        net (gluon.Block): 全局模型。
        lr (float): 学习率，占位参数。
        nfake (int): 恶意客户端数量。
        byz (callable): 拜占庭攻击函数。
        history (nd.NDArray): 历史聚合向量。
        fixed_rand (nd.NDArray): 固定随机方向。
        init_model (list[nd.NDArray]): 初始模型参数列表。
        last_50_model (list[nd.NDArray]): 最近 50 轮模型参数。
        last_grad (nd.NDArray): 上一轮梯度。
        sf (float): 缩放因子。
        e (int): 当前轮次。

    返回:
        tuple[list[nd.NDArray], float]: 攻击后的梯度列表及更新后缩放因子。

    异常:
        无。

    复杂度:
        时间 O(d·n log n); 空间 O(d)，其中 d 为参数维度，n 为客户端数。

    费曼学习法:
        (A) 功能: 将恶意梯度篡改后对全部梯度逐维取中位数，更新全局模型。
        (B) 类比: 像收集多份测量结果，先考虑作弊者，再对每个维度取中间值确保稳健。
        (C) 步骤拆解:
            1. 将每个客户端梯度展平成列向量，便于对齐维度。
            2. 调用攻击函数对前 nfake 个梯度进行潜在篡改。
            3. 对 NaN/Inf 进行替换，避免影响排序。
            4. 当客户端很多时按块处理取中位数，否则直接排序取中位。
            5. 将聚合结果拆分回原参数形状并写回模型。
        (D) 示例:
            - 调用: `median(gradients, net, 0.01, 5, byz, hist, rand, init, last50, last_grad, 2., 10)`
            - 输出: 更新后的梯度列表与缩放因子。
        (E) 边界与测试: 要求梯度维度一致；可测试极端异常值是否被抑制。
        (F) 背景与参考: 参见《Byzantine-robust Distributed Learning》关于中值聚合的分析。
    """
    param_list = [nd.concat(*[xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]
    fang_attack = getattr(byzantine, "fang_attack", None)
    opt_fang = getattr(byzantine, "opt_fang", None)
    if byz in (fang_attack, opt_fang):
        param_list, sf = byz(param_list, net, lr, nfake, history, fixed_rand, init_model, last_50_model, last_grad, e, sf, "median")
    else:
        param_list, sf = byz(param_list, net, lr, nfake, history, fixed_rand, init_model, last_50_model, last_grad, e, sf)
    for i, param in enumerate(param_list):
        # 将非法数值替换为大常数，避免排序出错
        mask = mx.nd.contrib.isnan(param) + mx.nd.contrib.isinf(param)
        param_list[i] = mx.nd.where(mask, mx.nd.ones_like(param) * 100000, param)

    if len(param_list) >= 100:
        block_size = 10000
        block_wise_nd = nd.concat(*param_list, dim=1)
        global_update = nd.zeros(param_list[0].size)
        for i in range(global_update.size // block_size):
            global_update[i * block_size:(i + 1) * block_size] = block_wise_median(block_wise_nd[i * block_size:(i + 1) * block_size])
        global_update[global_update.size // block_size * block_size:] = block_wise_median(block_wise_nd[global_update.size // block_size * block_size:])
    else:
        sorted_array = nd.sort(nd.concat(*param_list, dim=1), axis=-1)
        if sorted_array.shape[-1] % 2 == 1:
            global_update = sorted_array[:, int(sorted_array.shape[-1] / 2)]
        else:
            global_update = (sorted_array[:, int(sorted_array.shape[-1] / 2 - 1)] + sorted_array[:, int(sorted_array.shape[-1] / 2)]) / 2

    global_update.wait_to_read()
    # 写回全局模型参数
    idx = 0
    for param in net.collect_params().values():
        size = param.data().size
        param.set_data(param.data() + global_update[idx:idx+size].reshape(param.data().shape))
        idx += size
    return param_list, sf


def simple_mean(gradients, net, lr, nfake, byz, history, fixed_rand, init_model, last_50_model, last_grad, sf, e):
    """
    简要概述: 对梯度执行普通平均前先允许攻击函数写入恶意梯度。

    参数:
        gradients (list[list[nd.NDArray]]): 客户端梯度集合。
        net (gluon.Block): 全局模型。
        lr (float): 学习率，占位参数。
        nfake (int): 恶意客户端数量。
        byz (callable): 攻击函数。
        history (nd.NDArray): 历史聚合向量。
        fixed_rand (nd.NDArray): 固定随机方向。
        init_model (list[nd.NDArray]): 初始模型参数。
        last_50_model (list[nd.NDArray]): 最近模型快照。
        last_grad (nd.NDArray): 上一轮梯度。
        sf (float): 缩放因子。
        e (int): 当前轮次。

    返回:
        tuple[list[nd.NDArray], float]: 攻击后的梯度列表和缩放因子。

    异常:
        无。

    复杂度:
        时间 O(d·n); 空间 O(d)。

    费曼学习法:
        (A) 功能: 作为基线，将恶意梯度注入后直接求平均。
        (B) 类比: 像把所有投票分数相加取平均，不做额外防护。
        (C) 步骤拆解:
            1. 将梯度展平统一维度并交给攻击函数可能修改。
            2. 直接对所有列求均值，得到全局更新。
            3. 将更新量写回模型参数。
        (D) 示例:
            - 调用: `simple_mean(gradients, net, 0.01, 5, byz, hist, rand, init, last50, last_grad, 2., 10)`
            - 输出: 更新后的梯度列表和缩放因子。
        (E) 边界与测试: 易受极端值影响；可验证攻击存在时聚合的偏移程度。
        (F) 背景与参考: 对应 FedAvg 的聚合方式。
    """
    param_list = [nd.concat(*[xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]
    param_list, sf = byz(param_list, net, lr, nfake, history, fixed_rand, init_model, last_50_model, last_grad, e, sf)
    global_update = nd.mean(nd.concat(*param_list, dim=1), axis=-1)
    idx = 0
    for param in net.collect_params().values():
        size = param.data().size
        param.set_data(param.data() + global_update[idx:idx+size].reshape(param.data().shape))
        idx += size
    return param_list, sf


def mean_norm(gradients, net, lr, nfake, byz, history, fixed_rand, init_model, last_50_model, last_grad, sf, e):
    """
    简要概述: 对梯度执行范数裁剪后求平均，限制恶意梯度能量。

    参数:
        gradients (list[list[nd.NDArray]]): 客户端梯度列表。
        net (gluon.Block): 全局模型。
        lr (float): 学习率，占位参数。
        nfake (int): 恶意客户端数量。
        byz (callable): 攻击函数。
        history (nd.NDArray): 历史聚合向量。
        fixed_rand (nd.NDArray): 固定随机方向。
        init_model (list[nd.NDArray]): 初始模型参数。
        last_50_model (list[nd.NDArray]): 最近模型快照。
        last_grad (nd.NDArray): 上一轮梯度。
        sf (float): 缩放因子。
        e (int): 当前轮次。

    返回:
        tuple[nd.NDArray, float]: 归一化后的梯度矩阵与缩放因子。

    异常:
        无。

    复杂度:
        时间 O(d·n); 空间 O(d)。

    费曼学习法:
        (A) 功能: 在聚合前将每个梯度裁剪到安全范数，再求平均。
        (B) 类比: 设定最大音量，超出的声音自动调低，最后合成更均衡。
        (C) 步骤拆解:
            1. 展平梯度并让攻击函数注入恶意项。
            2. 计算各列范数与良性梯度平均范数比较。
            3. 对超阈值列按比例缩放。
            4. 对裁剪后的列求平均并写回模型。
        (D) 示例:
            - 调用: `mean_norm(gradients, net, 0.01, 5, byz, hist, rand, init, last50, last_grad, 2., 10)`
            - 输出: 范数裁剪后的聚合梯度和缩放因子。
        (E) 边界与测试: 要求 `len(param_norms[0]) > nfake`; 可测试极端范数时裁剪效果。
        (F) 背景与参考: 与 DP-SGD 等梯度裁剪策略相似。
    """
    param_list = [nd.concat(*[xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]
    param_list, sf = byz(param_list, net, lr, nfake, history, fixed_rand, init_model, last_50_model, last_grad, e, sf)

    param_list = nd.concat(*param_list, dim=1)
    param_norms = nd.norm(param_list, axis=0, keepdims=True)
    nb = nd.sum(param_norms[0, nfake:]) / (len(param_norms[0]) - nfake)
    # 对每列按阈值缩放，限制恶意能量
    param_list = param_list * nd.minimum(param_norms + 1e-7, nb) / (param_norms + 1e-7)
    global_update = nd.mean(param_list, axis=-1)
    idx = 0
    for param in net.collect_params().values():
        size = param.data().size
        param.set_data(param.data() + global_update[idx:idx+size].reshape(param.data().shape))
        idx += size
    return param_list, sf


def score(gradient, v, nfake):
    """
    简要概述: 计算目标梯度与其邻近梯度之间的距离和，用于异常检查。

    参数:
        gradient (nd.NDArray): 目标梯度。
        v (nd.NDArray): 所有梯度矩阵。
        nfake (int): 恶意客户端估计数量。

    返回:
        float: 最近邻距离之和。

    异常:
        无。

    复杂度:
        时间 O(n d log n); 空间 O(n)。

    费曼学习法:
        (A) 功能: 衡量某个梯度与周围梯度的相似度，得分越小越可信。
        (B) 类比: 看一个学生的答案与周围同学是否接近。
        (C) 步骤拆解:
            1. 计算目标与所有梯度的平方距离。
            2. 排序后剔除自身和恶意客户端数量对应的项。
            3. 对剩余最近的若干距离求和。
        (D) 示例:
            - 调用: `score(nd.array([1,1]), nd.array([[1,1],[2,2],[10,10]]), 1)`
            - 输出: 得到一个浮点得分。
        (E) 边界与测试: 要求列数大于 `nfake + 1`; 可测试有无异常值时分数差异。
        (F) 背景与参考: 与 Krum 等鲁棒聚合中的评分机制类似。
    """
    num_neighbours = v.shape[1] - 2 - nfake
    sorted_distance = nd.square(v - gradient).sum(axis=0).sort()
    return nd.sum(sorted_distance[1:(1+num_neighbours)]).asscalar()


def nearest_distance(gradient, c_p):
    """
    简要概述: 计算梯度与最近聚类中心之间的距离。

    参数:
        gradient (nd.NDArray): 目标梯度。
        c_p (nd.NDArray): 聚类中心集合。

    返回:
        float: 到最近中心的平方距离。

    异常:
        无。

    复杂度:
        时间 O(k d log k); 空间 O(k)。

    费曼学习法:
        (A) 功能: 找出距离目标梯度最近的中心点并量化距离。
        (B) 类比: 找到离自己最近的城市，距离越大越偏远。
        (C) 步骤拆解:
            1. 计算目标梯度与每个中心的平方距离。
            2. 对距离排序，获取次小的一个（排除自身）。
            3. 返回该距离作为指标。
        (D) 示例:
            - 调用: `nearest_distance(nd.array([1,1]), nd.array([[0,0],[1,1],[5,5]]))`
            - 输出: 接近 0。
        (E) 边界与测试: 需要至少两个中心；可测试目标即为其中一个中心的情形。
        (F) 背景与参考: 与聚类分析中常用的距离度量一致。
    """
    sorted_distance = nd.square(c_p - gradient).sum(axis=1).sort(axis=0)
    return sorted_distance[1].asscalar()


def score_gmm(gradient, v, nfake):
    """
    简要概述: 计算目标梯度与最近 nfake-1 个邻居的距离和，用于聚类过滤。

    参数:
        gradient (nd.NDArray): 目标梯度。
        v (nd.NDArray): 梯度矩阵。
        nfake (int): 恶意客户端数量。

    返回:
        float: 最近邻距离之和。

    异常:
        无。

    复杂度:
        时间 O(n d log n); 空间 O(n)。

    费曼学习法:
        (A) 功能: 找出与目标梯度最接近的若干邻居，衡量其是否成团。
        (B) 类比: 选出和某人最合拍的朋友，观察是否抱团行动。
        (C) 步骤拆解:
            1. 计算目标与每个梯度的平方距离。
            2. 排序后选取从第二个元素开始的 `nfake-1` 个。
            3. 对这些距离求和得到评分。
        (D) 示例:
            - 调用: `score_gmm(nd.array([1,1]), nd.array([[1,1],[1.1,1.1],[10,10]]), 2)`
            - 输出: 得到较小的距离值。
        (E) 边界与测试: 要求 `nfake >= 1`；可对比无异常和有异常的得分。
        (F) 背景与参考: 与基于聚类的鲁棒聚合策略相关。
    """
    num_neighbours = nfake - 1
    sorted_distance = nd.square(v - gradient).sum(axis=0).sort()
    return nd.sum(sorted_distance[1:(1+num_neighbours)]).asscalar()


def trim(gradients, net, lr, nfake, byz, history, fixed_rand, init_model, last_50_model, last_grad, sf, e):
    """
    简要概述: 执行截断均值聚合，并允许拜占庭攻击在聚合前写入恶意梯度。

    参数:
        gradients (list[list[nd.NDArray]]): 客户端梯度集合。
        net (gluon.Block): 全局模型。
        lr (float): 学习率，占位参数。
        nfake (int): 恶意客户端数量。
        byz (callable): 攻击函数。
        history (nd.NDArray): 历史聚合向量。
        fixed_rand (nd.NDArray): 固定随机方向。
        init_model (list[nd.NDArray]): 初始模型参数。
        last_50_model (list[nd.NDArray]): 最近模型快照。
        last_grad (nd.NDArray): 上一轮梯度。
        sf (float): 缩放因子。
        e (int): 当前轮次。

    返回:
        tuple[list[nd.NDArray] | int, float]: 成功时返回聚合前梯度列表与缩放因子，若客户端数不足则返回 -1。

    异常:
        无。

    复杂度:
        时间 O(d·n log n); 空间 O(d)。

    费曼学习法:
        (A) 功能: 在恶意梯度注入后，去掉最大和最小的若干列再平均。
        (B) 类比: 对多名裁判的分数去掉最高最低分，再求平均。
        (C) 步骤拆解:
            1. 展平梯度并给攻击函数机会篡改。
            2. 统计每列是否含非法数值，并进行替换。
            3. 当客户端很多时按块执行截断均值，否则直接排序求均值。
            4. 将结果写回模型。
        (D) 示例:
            - 调用: `trim(gradients, net, 0.01, 5, byz, hist, rand, init, last50, last_grad, 2., 10)`
            - 输出: 聚合后梯度列表与缩放因子。
        (E) 边界与测试: 若 `n <= 2*nfake` 将返回 -1；可测试极端恶意值时能否剔除。
        (F) 背景与参考: 参见《Machine Learning with Adversaries》对截断均值的讨论。
    """
    param_list = [nd.concat(*[xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]
    fang_attack = getattr(byzantine, "fang_attack", None)
    opt_fang = getattr(byzantine, "opt_fang", None)
    if byz in (fang_attack, opt_fang):
        param_list, sf = byz(param_list, net, lr, nfake, history, fixed_rand, init_model, last_50_model, last_grad, e, sf, "trim")
    else:
        param_list, sf = byz(param_list, net, lr, nfake, history, fixed_rand, init_model, last_50_model, last_grad, e, sf)
    b = nfake
    n = len(param_list)
    m = n - 2 * b
    for i, param in enumerate(param_list):
        mask = mx.nd.contrib.isnan(param) + mx.nd.contrib.isinf(param)
        param_list[i] = mx.nd.where(mask, mx.nd.ones_like(param) * 100000, param)

    if m <= 0:
        return -1

    if len(param_list) >= 100:
        block_size = 10000
        block_wise_nd = nd.concat(*param_list, dim=1)
        global_update = nd.zeros(param_list[0].size)
        for i in range(global_update.size // block_size):
            global_update[i * block_size:(i + 1) * block_size] = block_wise_trim(block_wise_nd[i * block_size:(i + 1) * block_size], b, m)
        global_update[global_update.size // block_size * block_size:] = block_wise_trim(block_wise_nd[global_update.size // block_size * block_size:], b, m)
    else:
        sorted_array = nd.sort(nd.concat(*param_list, dim=1), axis=-1)
        global_update = nd.mean(sorted_array[:, b:(b+m)], axis=-1)

    idx = 0
    for param in net.collect_params().values():
        size = param.data().size
        param.set_data(param.data() + global_update[idx:idx+size].reshape(param.data().shape))
        idx += size

    return param_list, sf


__AI_ANNOTATION_SUMMARY__ = """
block_wise_median: 对梯度分块取中位数以抑制极端值。
block_wise_trim: 剪除两端后求均值，降低恶意梯度影响。
cos_sim_nd: 计算向量余弦距离评估方向差异。
median: 调用攻击函数后执行中值聚合。
simple_mean: 恶意注入后直接求算术平均。
mean_norm: 对梯度裁剪范数再求均值。
score: 通过邻域距离衡量梯度可信度。
nearest_distance: 计算梯度到最近中心的距离。
score_gmm: 统计局部邻居距离用于聚类过滤。
trim: 执行截断均值聚合更新全局模型。
"""
