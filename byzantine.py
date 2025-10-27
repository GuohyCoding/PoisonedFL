"""
byzantine.py
=====================
该模块汇总了多种联邦学习中的拜占庭攻击策略与无攻击占位实现，
旨在帮助研究者模拟恶意客户端如何在模型聚合阶段注入异常梯度。
"""

import numpy as np
from mxnet import nd, autograd, gluon


def no_byz(v, net, lr, nfake, history, fixed_rand, init_model, last_50_model, last_grad, e, scaling_factor=100000.):
    """
    简要概述: 无拜占庭攻击的占位实现，直接返回原始更新与缩放因子。

    参数:
        v (nd.NDArray): 当前收集到的客户端更新矩阵，形状通常为 (客户端数, 参数维度)。
        net (gluon.Block): 服务器端的全局模型，仅为保持接口一致性。
        lr (float): 学习率，占位参数。
        nfake (int): 恶意客户端数量，占位参数。
        history (nd.NDArray): 历史聚合梯度，占位参数。
        fixed_rand (nd.NDArray): 固定随机方向，占位参数。
        init_model (list[nd.NDArray]): 初始模型参数列表，占位参数。
        last_50_model (list[nd.NDArray]): 最近 50 轮模型快照，占位参数。
        last_grad (nd.NDArray): 上一轮梯度，占位参数。
        e (int): 当前迭代轮次，占位参数。
        scaling_factor (float): 预设缩放因子，默认 100000.

    返回:
        tuple[nd.NDArray, float]: 原样返回的更新矩阵及缩放因子。

    异常:
        无。

    复杂度:
        时间 O(1); 空间 O(1)。

    费曼学习法:
        (A) 功能: 该函数完全不修改输入，让管线在无攻击情况下顺利运行。
        (B) 类比: 好比透明胶片，盖在图纸上不会遮挡任何内容。
        (C) 步骤拆解:
            1. 接收所有参数，为保持接口统一但不做实际计算。
            2. 返回输入梯度矩阵与缩放因子本身。
        (D) 示例:
            - 调用: `v, sf = no_byz(v, net, 0.01, 0, hist, rand, init, last50, last_grad, 1)`
            - 输出: `v` 与 `sf` 完全等于传入值。
        (E) 边界与测试: 只要传入对象满足接口即可；可测试对象身份是否保持不变。
        (F) 背景与参考: 用于构建鲁棒聚合的对照组，可参考经典 FedAvg 设置。
    """
    return v, scaling_factor


def compute_lambda(all_updates, model_re, n_attackers):
    """
    简要概述: 基于客户端更新之间的几何关系估计安全缩放界 λ。

    参数:
        all_updates (nd.NDArray): 所有客户端梯度组成的矩阵，形状 (客户端数, 参数维度)。
        model_re (nd.NDArray): 当前全局模型的扁平化向量表示。
        n_attackers (int): 预估的恶意客户端数量。

    返回:
        nd.NDArray: 单元素张量，表示建议的缩放强度 λ。

    异常:
        无。

    复杂度:
        时间 O(n^2 d); 空间 O(n d)，其中 n 为客户端数，d 为参数维度。

    费曼学习法:
        (A) 功能: 计算一个兼顾整体偏移与局部一致性的最大安全偏移量。
        (B) 类比: 像合照时测量每个人离中心的距离，从而判断谁站得太偏。
        (C) 步骤拆解:
            1. 遍历每位客户端，计算它与其他客户端更新之间的欧氏距离。
            2. 将每行距离排序，通过忽略最远的推测攻击者获得稳健的距离总和。
            3. 取距离总和的最小值并按维度归一化，形成 term_1。
            4. 额外计算所有更新到参考模型的最大距离，捕捉整体偏移度。
            5. 将两部分相加作为 λ，确保攻击不会过于突兀。
        (D) 示例:
            - 调用: `lam = compute_lambda(nd.random.normal(shape=(10,5)), nd.random.normal(shape=(5,)), 2)`
            - 输出: `lam` 为单个标量 NDArray，代表缩放上限。
        (E) 边界与测试: 要求 n_attackers < n_benign-1；建议测试无攻击和高攻击比例两种极端情况。
        (F) 背景与参考: 与几何中值和鲁棒聚合理论相关，可参考《Byzantine-robust Distributed Learning: Towards Optimal Statistical Rates》。
    """
    distances = []
    n_benign, d = all_updates.shape
    for update in all_updates:
        # 计算当前客户端与其余客户端的 L2 距离
        distance = nd.norm(all_updates - update, axis=1)
        distances.append(distance)
    # 堆叠成距离矩阵以便后续排序与截断
    distances = nd.stack(*distances)

    # 排序后截断潜在攻击者对计分的影响
    distances = nd.sort(distances, axis=1)
    scores = nd.sum(distances[:, :n_benign - 1 - n_attackers], axis=1)
    min_score = nd.min(scores)
    term_1 = min_score / ((n_benign - n_attackers - 1)
                          * nd.sqrt(nd.array([d]))[0])
    max_wre_dist = nd.max(nd.norm(all_updates - model_re,
                          axis=1)) / (nd.sqrt(nd.array([d]))[0])
    return (term_1 + max_wre_dist)


def score(gradient, v, nbyz):
    """
    简要概述: 计算梯度与邻域内其他梯度的距离得分，衡量是否可疑。

    参数:
        gradient (nd.NDArray): 目标客户端的梯度向量。
        v (nd.NDArray): 所有客户端梯度构成的矩阵。
        nbyz (int): 预估的恶意客户端数量。

    返回:
        float: 排除恶意节点后最近邻距离之和。

    异常:
        无。

    复杂度:
        时间 O(n d log n); 空间 O(n)，其中 n 为客户端数，d 为参数维度。

    费曼学习法:
        (A) 功能: 给每个梯度打一个“偏离程度”分数，数值越小越可信。
        (B) 类比: 将作业成绩排序后剔除异常值，计算剩余同学与目标的差距。
        (C) 步骤拆解:
            1. 计算目标梯度与所有梯度的平方差。
            2. 将差值排序，排除自身和推测的恶意节点。
            3. 对剩下最近的若干差值求和，得到得分。
        (D) 示例:
            - 调用: `val = score(nd.array([1,2]), nd.array([[1,2],[2,3],[10,10]]), 1)`
            - 输出: `val` 为浮点数，越小表示越可信。
        (E) 边界与测试: 要保证 `nbyz < v.shape[0]-2`；建议构造一组含明显异常点的测试案例验证排序与截断逻辑。
        (F) 背景与参考: 与 Krum 等鲁棒聚合算法中的近邻评分思想一致，可参考《The Krum Algorithm》。
    """
    num_neighbours = v.shape[0] - 2 - nbyz
    sorted_distance = nd.square(v - gradient).sum(axis=1).sort()
    return nd.sum(sorted_distance[1:(1+num_neighbours)]).asscalar()


def poisonedfl(v, net, lr, nfake, history, fixed_rand, init_model, last_50_model, last_grad, e, scaling_factor=100000.):
    """
    简要概述: 实现 PoisonedFL 攻击策略，根据历史方向生成针对性恶意更新。

    参数:
        v (nd.NDArray): 所有客户端的更新矩阵，第一维索引客户端。
        net (gluon.Block): 当前全局模型实例。
        lr (float): 学习率，占位参数。
        nfake (int): 恶意客户端数量。
        history (nd.NDArray): 历史聚合梯度向量。
        fixed_rand (nd.NDArray): 固定随机方向，用于保持攻击一致性。
        init_model (list[nd.NDArray]): 初始模型参数列表。
        last_50_model (list[nd.NDArray]): 过去 50 轮的模型快照。
        last_grad (nd.NDArray): 上一轮全局梯度。
        e (int): 当前迭代轮次。
        scaling_factor (float): 攻击缩放因子，默认 100000.

    返回:
        tuple[nd.NDArray, float]: 修改后的梯度矩阵和可能调整后的缩放因子。

    异常:
        NotImplementedError: 当 `fixed_rand` 的维度不在预设表中时抛出。

    复杂度:
        时间 O(d + nfake·d); 空间 O(d)，d 为参数总维度。

    费曼学习法:
        (A) 功能: 让恶意客户端沿固定方向推送大幅更新，误导整体模型。
        (B) 类比: 如同在拔河比赛中偷偷安排几人按既定节奏用力，使绳子偏向特定方向。
        (C) 步骤拆解:
            1. 根据参数维度选择经验阈值 `k_95` 和 `k_99`，用来判断攻击与固定方向的对齐程度。
            2. 复制服务器当前参数，计算历史梯度和上一轮梯度的范数，用于规范化。
            3. 通过投影剔除上一轮梯度方向，获得残差方向并与固定随机向量对齐。
            4. 若迭代轮次满足 50 的倍数，统计模型增量与随机方向的符号一致度，调整缩放因子避免过度暴露。
            5. 使用 λ 放大偏移向量，将结果写入前 `nfake` 个客户端条目作为恶意更新。
        (D) 示例:
            - 调用: `v, sf = poisonedfl(v, net, 0.01, 5, hist, rand_vec, init, last50, last_grad, 100, 10.)`
            - 输出: 返回的 `v` 前五行被恶意更新覆盖，`sf` 可能被自适应缩放。
        (E) 边界与测试: `fixed_rand` 需与模型维度一致；若历史为空需预置 NDArray；建议测试在 `aligned_dim_cnt` 接近阈值时缩放因子是否变化。
        (F) 背景与参考: 与模型投毒相关，可参考《Model Poisoning Attacks against Federated Learning》及《Manipulating Byzantine-robust Aggregation》。
    """
    # 针对不同模型维度预先设定二项分布阈值 k_95 与 k_99
    if fixed_rand.shape[0] == 1204682:
        k_95 = 603244
        k_99 = 603618
    elif fixed_rand.shape[0] == 139960:
        k_95 = 70288
        k_99 = 70415
    elif fixed_rand.shape[0] == 717924:
        k_95 = 359659
        k_99 = 359948
    elif fixed_rand.shape[0] == 145212:
        k_95 = 72919
        k_99 = 73049
    else:
        raise NotImplementedError
    sf = scaling_factor

    # 攻击从第 2 轮开始启用，需要依赖历史梯度
    if isinstance(history, nd.NDArray):
        # 复制当前模型参数，避免直接修改原始对象
        current_model = [param.data().copy() for param in net.collect_params().values()]
        history_norm = nd.norm(history)
        last_grad_norm = nd.norm(last_grad)
        # 计算去除上一轮梯度后，历史方向的残差范数
        scale = nd.norm(history - nd.expand_dims(last_grad, axis=-1)* history_norm/(last_grad_norm+1e-9), axis=1)
        # 使用固定随机向量得到最终攻击方向
        deviation = scale * fixed_rand / (nd.norm(scale)+1e-9)

        # 动态计算缩放因子 lambda 以平衡隐蔽性与攻击强度
        if e % 50 == 0:
            # 统计与 50 轮前模型的整体变化，用于检测方向对齐度
            total_update = nd.concat(*[xx.reshape((-1, 1)) for xx in current_model],
                                dim=0) - nd.concat(*[xx.reshape((-1, 1)) for xx in last_50_model], dim=0)
            # 若存在零增量，使用当前模型参数替代以避免除零
            total_update = nd.where(total_update == 0, nd.concat(*[xx.reshape((-1, 1)) for xx in current_model],dim=0), total_update)
            current_sign = nd.sign(total_update)
            # 统计与固定随机方向符号一致的维度数，判断攻击是否显眼
            aligned_dim_cnt = (current_sign == nd.expand_dims(fixed_rand, axis=-1)).sum()
            if aligned_dim_cnt < k_99 and scaling_factor*0.7>=0.5:
                sf = scaling_factor*0.7
            else:
                sf = scaling_factor
            lamda_succ = sf * history_norm
        else:
            sf = scaling_factor
            lamda_succ = sf * history_norm
        mal_update = lamda_succ * deviation
        for i in range(nfake):
            # 将同一恶意向量广播给所有攻击客户端
            v[i] = nd.expand_dims(mal_update, axis=-1)
    return v, sf


def random_attack(v, net, lr, nfake, history, fixed_rand,  init_model, last_50_model, last_grad,e, scaling_factor=100000.):
    """
    简要概述: 通过高斯噪声生成恶意更新，以随机扰乱聚合方向。

    参数:
        v (nd.NDArray): 客户端梯度矩阵。
        net (gluon.Block): 当前全局模型，占位参数。
        lr (float): 学习率，占位参数。
        nfake (int): 恶意客户端数量。
        history (nd.NDArray): 历史梯度，占位参数。
        fixed_rand (nd.NDArray): 固定随机方向，占位参数。
        init_model (list[nd.NDArray]): 初始模型参数，占位参数。
        last_50_model (list[nd.NDArray]): 最近 50 轮模型，占位参数。
        last_grad (nd.NDArray): 上一轮梯度，占位参数。
        e (int): 当前轮次，占位参数。
        scaling_factor (float): 控制噪声幅度的缩放因子。

    返回:
        tuple[nd.NDArray, float]: 更新后的梯度矩阵与缩放因子。

    异常:
        无。

    复杂度:
        时间 O(nfake·d); 空间 O(1) 额外。

    费曼学习法:
        (A) 功能: 给恶意节点填入放大的高斯噪声，让聚合难以收敛。
        (B) 类比: 像在问卷调查中胡乱填写大量随机答案，扰乱统计结果。
        (C) 步骤拆解:
            1. 遍历每个恶意客户端索引。
            2. 按目标梯度形状采样高斯噪声并乘缩放因子放大。
            3. 将噪声写入 `v` 的对应条目。
        (D) 示例:
            - 调用: `v, sf = random_attack(v, net, 0.01, 3, hist, rand, init, last50, last_grad, 10, 5.)`
            - 输出: `v` 的前三行被随机噪声覆盖，`sf` 为 5.0。
        (E) 边界与测试: 需保证 `v` 行数 ≥ `nfake`；可通过固定随机种子测试噪声分布。
        (F) 背景与参考: 对应经典高斯投毒策略，可参考《How Byzantine Robustness Can Handle Diversity》。
    """
    for i in range(nfake):
        # 生成与原梯度同形状的高斯噪声并放大影响力
        v[i] = scaling_factor * nd.random.normal(loc=0, scale=1, shape=v[0].shape)
    return v, scaling_factor


def init_attack(v, net, lr, nfake, history, fixed_rand, init_model, last_50_model, last_grad,e, scaling_factor=100000.):
    """
    简要概述: 构造指向初始模型的恶意更新，使训练进展被“拖回起点”。

    参数:
        v (nd.NDArray): 客户端梯度矩阵。
        net (gluon.Block): 当前全局模型。
        lr (float): 学习率，占位参数。
        nfake (int): 恶意客户端数量。
        history (nd.NDArray): 历史梯度，占位参数。
        fixed_rand (nd.NDArray): 固定随机方向，占位参数。
        init_model (list[nd.NDArray]): 初始模型参数列表。
        last_50_model (list[nd.NDArray]): 最近 50 轮模型，占位参数。
        last_grad (nd.NDArray): 上一轮梯度，占位参数。
        e (int): 当前轮次，占位参数。
        scaling_factor (float): 控制恶意方向强度的缩放因子。

    返回:
        tuple[nd.NDArray, float]: 修改后的梯度矩阵与缩放因子。

    异常:
        无。

    复杂度:
        时间 O(d + nfake·d); 空间 O(d)。

    费曼学习法:
        (A) 功能: 让恶意客户端沿初始模型方向发力，抵消正向训练。
        (B) 类比: 就像有人在队伍前进时不断往回拉，迫使大家停步。
        (C) 步骤拆解:
            1. 复制当前模型参数，避免直接原地修改。
            2. 拉平成向量后计算初始模型与当前模型的差值方向。
            3. 将差值乘以缩放因子，写入前 `nfake` 个客户端条目。
        (D) 示例:
            - 调用: `v, sf = init_attack(v, net, 0.01, 2, hist, rand, init, last50, last_grad, 20, 3.)`
            - 输出: `v` 前两行朝初始方向偏移，`sf` 为 3.0。
        (E) 边界与测试: 需要保证 `init_model` 与当前模型维度一致；建议测试在训练初期与后期的拖拽效果差异。
        (F) 背景与参考: 与模型回滚攻击相关，可参考《Backdoor Attacks on Federated Learning》。
    """
    # 复制当前模型参数，避免对原对象造成副作用
    current_model = [param.data().copy() for param in net.collect_params().values()]
    # 计算初始模型相对当前模型的方向
    direction = nd.concat(*[xx.reshape((-1, 1)) for xx in init_model], dim=0) - nd.concat(*[xx.reshape((-1, 1)) for xx in current_model], dim=0)
    for i in range(nfake):
        # 将相同方向复制到所有恶意客户端槽位
        v[i] = scaling_factor * direction
    return v, scaling_factor


__AI_ANNOTATION_SUMMARY__ = """
no_byz: 无攻击占位实现，保持梯度与缩放因子不变返回。
compute_lambda: 利用客户端间距估计安全缩放上界 λ。
score: 通过近邻距离为梯度可信度打分。
poisonedfl: 沿固定随机方向构造放大的恶意更新误导聚合。
random_attack: 向恶意客户端注入高斯噪声扰乱训练。
init_attack: 沿初始模型方向拖拽训练进程的投毒策略。
"""
