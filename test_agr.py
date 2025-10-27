"""
test_agr.py
===========
该模块承担联邦学习实验主逻辑，涵盖数据加载、模型构建、聚合策略选择
以及攻击注入，为复现实验提供脚本入口。
"""

from __future__ import print_function
import torch.nn.functional as F
import torch.nn as nn
import nd_aggregation
import mxnet as mx
from mxnet import nd, autograd, gluon
import numpy as np
import random
import argparse
import byzantine
import seaborn as sns
import matplotlib.pyplot as plt
import os

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from utils import *
import torch
def parse_args():
    """
    简要概述: 构建并解析联邦学习实验的命令行参数集合。

    参数:
        无。

    返回:
        argparse.Namespace: 包含所有配置项的命名空间对象。

    异常:
        无。

    复杂度:
        时间 O(k); 空间 O(1)，其中 k 为参数数量。

    费曼学习法:
        (A) 功能: 定义实验脚本可以接受的命令行参数并返回解析结果。
        (B) 类比: 类似填写报名表格，列出每个字段的名称和默认值，最后拿到填好的表。
        (C) 步骤拆解:
            1. 创建 `ArgumentParser` 对象用于收集参数。
            2. 逐个注册实验需要的参数、类型及默认值。
            3. 调用 `parse_args` 从命令行读取真实输入并生成结果对象。
        (D) 示例:
            - 调用: `args = parse_args()`
            - 输出: `args.dataset` 等属性可直接访问配置。
        (E) 边界与测试: 需避免重复参数名；可编写单元测试模拟命令行列表验证解析。
        (F) 背景与参考: 基于 Python 标准库 argparse，可参考《Python 官方文档 argparse》。
    """
    parser = argparse.ArgumentParser()
    # 注册联邦学习实验所需的全部命令行参数
    parser.add_argument("--server_pc", help="the number of data the server holds", type=int, default=100)
    parser.add_argument("--dataset", help="dataset", type=str, default="FashionMNIST")
    parser.add_argument("--bias", help="degree of non-iid", type=float, default=0.5)
    parser.add_argument("--net", help="net", type=str, default="cnn")
    parser.add_argument("--batch_size", help="batch size", type=int, default=32)
    parser.add_argument("--lr", help="learning rate", type=float, default=0.01)
    parser.add_argument("--nworkers", help="# workers", type=int, default=1200)
    parser.add_argument("--niter", help="# iterations", type=int, default=200)
    parser.add_argument("--gpu", help="index of gpu", type=int, default=0)
    parser.add_argument("--seed", help="seed", type=int, default=42)
    parser.add_argument("--selected_layer", help="selected_layer", type=int, default=0)
    parser.add_argument("--nfake", help="# fake clients", type=int, default=100)
    parser.add_argument("--byz_type", help="type of attack", type=str, default="no")
    parser.add_argument("--aggregation", help="aggregation", type=str, default="fltrust")
    parser.add_argument("--p", help="bias probability of 1 in server sample", type=float, default=0.1)
    parser.add_argument("--sf", help="scaling factor", type=float, default=10)
    parser.add_argument("--participation_rate",help="participation_rate", type=float, default=0.025)
    parser.add_argument("--step", help="period to log accuracy", type=int, default=1000)
    parser.add_argument("--local_epoch", help="local_epoch", type=int, default=0)

    return parser.parse_args()

def get_device(device):
    """
    简要概述: 根据输入索引选择 MXNet 计算设备。

    参数:
        device (int): GPU 序号，若为 -1 表示使用 CPU。

    返回:
        mx.context.Context: MXNet 上下文对象，可用于张量运算。

    异常:
        无。

    复杂度:
        时间 O(1); 空间 O(1)。

    费曼学习法:
        (A) 功能: 决定计算应在 CPU 还是某块 GPU 上执行。
        (B) 类比: 像给工人分配工作台，-1 表示安排在公共桌，其他数字代表指定的机器。
        (C) 步骤拆解:
            1. 判断传入设备编号是否为 -1。
            2. 若为 -1，构造 CPU 上下文；否则构造指定 GPU 上下文。
            3. 返回该上下文供后续数据迁移或运算使用。
        (D) 示例:
            - 调用: `ctx = get_device(0)`
            - 输出: `ctx` 为 GPU(0)。
        (E) 边界与测试: 若系统无对应 GPU 会抛出运行时错误；建议测试 CPU 与 GPU 两种路径。
        (F) 背景与参考: MXNet 上下文管理，与官方文档 Context 部分相关。
    """
    if device == -1:
        ctx = mx.cpu()
    else:
        ctx = mx.gpu(device)
    return ctx
    
def get_dnn(num_outputs=600):
    """
    简要概述: 构建两层全连接的前馈神经网络。

    参数:
        num_outputs (int): 最后一层的输出维度，默认 600。

    返回:
        gluon.nn.Sequential: 已初始化的全连接网络结构。

    异常:
        无。

    复杂度:
        时间 O(1); 空间 O(1)（仅构图级别，不含训练）。

    费曼学习法:
        (A) 功能: 创建一个简单的前馈神经网络，方便后续绑定到训练流程。
        (B) 类比: 像搭建两层的积木塔，上层模块数量可调。
        (C) 步骤拆解:
            1. 初始化一个 `Sequential` 容器。
            2. 添加第一层 1024 单元、tanh 激活的全连接层。
            3. 添加输出为 `num_outputs` 的线性层。
        (D) 示例:
            - 调用: `net = get_dnn(100)`
            - 输出: `net` 是包含两层的网络结构。
        (E) 边界与测试: 需确保 `num_outputs > 0`；可通过前向传入随机张量测试输出形状。
        (F) 背景与参考: 属于基础 MLP 结构，可参考《Deep Learning》教材关于多层感知机的章节。
    """
    dnn = gluon.nn.Sequential()
    with dnn.name_scope():
        # 添加高维隐藏层，提升建模能力
        dnn.add(gluon.nn.Dense(1024, activation='tanh'))
        dnn.add(gluon.nn.Dense(num_outputs))
    return dnn

def get_cnn(num_outputs=10):
    """
    简要概述: 构建适用于 28x28 灰度图像的卷积神经网络。

    参数:
        num_outputs (int): 最后一层输出类数，默认 10。

    返回:
        gluon.nn.Sequential: 含两层卷积与全连接输出的 CNN。

    异常:
        无。

    复杂度:
        时间 O(1); 空间 O(1)（构图阶段）。

    费曼学习法:
        (A) 功能: 提供一个标准的小型 CNN 结构用于图像分类。
        (B) 类比: 像先用两种放大镜查看细节，再将信息打平送入决策器。
        (C) 步骤拆解:
            1. 创建顺序容器并进入命名作用域。
            2. 添加两组卷积 + 池化层提取空间特征。
            3. 扁平化特征后连接 100 单元和输出层完成分类。
        (D) 示例:
            - 调用: `net = get_cnn(10)`
            - 输出: `net` 可直接用于 FashionMNIST 训练。
        (E) 边界与测试: 输入假定为 1x28x28；可通过随机张量验证前向尺寸。
        (F) 背景与参考: 结构类似 LeNet，可查阅《Gradient-Based Learning Applied to Document Recognition》。
    """
    cnn = gluon.nn.Sequential()
    with cnn.name_scope():
        cnn.add(gluon.nn.Conv2D(channels=30, kernel_size=3, activation='relu'))
        cnn.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        cnn.add(gluon.nn.Conv2D(channels=50, kernel_size=3, activation='relu'))
        cnn.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        cnn.add(gluon.nn.Flatten())
        cnn.add(gluon.nn.Dense(100, activation="relu"))
        cnn.add(gluon.nn.Dense(num_outputs))
    return cnn


def get_cnn_cifar(num_outputs=10):
    """
    简要概述: 构建针对 CIFAR-10 彩色图像的卷积网络。

    参数:
        num_outputs (int): 输出类别数，默认 10。

    返回:
        gluon.nn.Sequential: 适配 3x32x32 输入的 CNN 模型。

    异常:
        无。

    复杂度:
        时间 O(1); 空间 O(1)（构图阶段）。

    费曼学习法:
        (A) 功能: 生成一个两层卷积 + 全连接结构的基线模型，适用于 CIFAR-10。
        (B) 类比: 类似使用两级滤镜处理彩色照片，再将特征交给分类器。
        (C) 步骤拆解:
            1. 创建顺序网络容器。
            2. 添加两次卷积-池化组合以提取多尺度特征。
            3. 展平后连接 512 单元及输出层完成分类。
        (D) 示例:
            - 调用: `net = get_cnn_cifar(10)`
            - 输出: `net` 可处理 CIFAR-10 图像。
        (E) 边界与测试: 输入须为 3x32x32；可通过一次随机前向测试尺寸。
        (F) 背景与参考: 结构借鉴简单 CNN 基线，可参阅《Deep Learning with Python》中相关示例。
    """
    cnn = gluon.nn.Sequential()
    with cnn.name_scope():
        cnn.add(gluon.nn.Conv2D(channels=32, kernel_size=3,in_channels=3, activation='relu'))
        cnn.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        cnn.add(gluon.nn.Conv2D(channels=64, kernel_size=3, activation='relu'))
        cnn.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        cnn.add(gluon.nn.Flatten())
        cnn.add(gluon.nn.Dense(512, activation="relu"))
        cnn.add(gluon.nn.Dense(num_outputs))
    return cnn



def get_net(net_type, num_outputs=10):
    """
    简要概述: 根据输入类型字符串构建对应的神经网络。

    参数:
        net_type (str): 网络类型标识，可选 'cnn'、'cnn_cifar'、'dnn'。
        num_outputs (int): 输出维度，默认 10。

    返回:
        gluon.Block: 组装好的神经网络实例。

    异常:
        NotImplementedError: 当给定类型不支持时抛出。

    复杂度:
        时间 O(1); 空间 O(1)。

    费曼学习法:
        (A) 功能: 根据配置自动选择合适的模型结构。
        (B) 类比: 像点餐时根据菜名选择不同的菜谱。
        (C) 步骤拆解:
            1. 判断输入类型标识。
            2. 调用相应的构建函数创建网络。
            3. 返回生成的网络供后续训练使用。
        (D) 示例:
            - 调用: `net = get_net('cnn', 10)`
            - 输出: `net` 为卷积网络。
        (E) 边界与测试: 输入类型必须在预设集合内；可通过多次调用验证不同返回结构。
        (F) 背景与参考: 提供模型工厂模式，可结合 Gluon 官方教程理解。
    """
    if net_type == 'cnn':
        net = get_cnn(num_outputs)
    elif net_type == "cnn_cifar":
        net = get_cnn_cifar(num_outputs)
    elif net_type == 'dnn':
        net = get_dnn(num_outputs)
    else:
        raise NotImplementedError
    return net


def get_shapes(dataset):
    """
    简要概述: 根据数据集名称返回输入形状、输出维度与标签数。

    参数:
        dataset (str): 数据集名称，如 'FashionMNIST'、'mnist'、'FEMNIST'、'cifar10'、'purchase'。

    返回:
        tuple[tuple, int, int]: 输入形状、输出维度、标签数量。

    异常:
        NotImplementedError: 当数据集名称不在支持列表中时。

    复杂度:
        时间 O(1); 空间 O(1)。

    费曼学习法:
        (A) 功能: 为不同数据集提供模型所需的输入输出尺寸配置。
        (B) 类比: 如同根据比赛项目决定器材规格，例如球类大小或赛道长度。
        (C) 步骤拆解:
            1. 匹配数据集名称。
            2. 为匹配项返回预设的输入形状与标签数。
            3. 若未匹配，抛出未实现异常提示扩展。
        (D) 示例:
            - 调用: `inputs, outputs, labels = get_shapes('cifar10')`
            - 输出: `inputs` 为 `(1, 3, 32, 32)`，`outputs` 和 `labels` 均为 10。
        (E) 边界与测试: 需保持 `args` 全局变量已定义；建议测试每个分支返回值正确。
        (F) 背景与参考: 依赖数据集标准尺寸，可参考各公开数据集描述。
    """
    if dataset == 'FashionMNIST' or dataset == 'mnist':
        num_inputs = (1, 1, 28, 28)
        num_outputs = 10
        num_labels = 10
    elif dataset == 'FEMNIST':
        num_inputs = (1, 1, 28, 28)
        num_outputs = 62
        num_labels = 62
    elif dataset == 'cifar10':
        num_inputs = (1, 3, 32, 32)
        num_outputs = 10
        num_labels = 10
    elif args.dataset == 'purchase':
        num_inputs = (1, 600)
        num_outputs = 100
        num_labels = 100
    else:
        raise NotImplementedError
    return num_inputs, num_outputs, num_labels

def evaluate_accuracy(data_iterator, net, ctx):
    """
    简要概述: 在指定上下文中评估模型在数据集上的准确率。

    参数:
        data_iterator (mx.gluon.data.DataLoader): 迭代提供 (data, label) 的数据迭代器。
        net (gluon.Block): 待评估的模型。
        ctx (mx.context.Context): 计算设备上下文。

    返回:
        float: 模型在提供数据上的分类准确率。

    异常:
        无。

    复杂度:
        时间 O(N·C); 空间 O(1)，其中 N 为样本数，C 为前向计算成本。

    费曼学习法:
        (A) 功能: 遍历数据集计算模型预测与真实标签的一致比例。
        (B) 类比: 像批改考试，逐份试卷判定正确与否，最终计算正确率。
        (C) 步骤拆解:
            1. 初始化 MXNet 的准确率评估器。
            2. 遍历数据迭代器，将数据与标签搬运到指定设备。
            3. 前向计算得到预测结果，取最大概率的类别。
            4. 更新准确率统计并最终返回平均值。
        (D) 示例:
            - 调用: `acc = evaluate_accuracy(test_loader, net, mx.cpu())`
            - 输出: `acc` 为 0~1 之间的浮点数。
        (E) 边界与测试: 需确保模型已经初始化；可通过构造小批量模拟器验证。
        (F) 背景与参考: 使用 MXNet `Accuracy` 指标，可参考 Gluon 官方评估器文档。
    """
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(data_iterator):
        # 将批次数据搬运到指定设备，保证预测与标签在同一上下文
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        remaining_idx = list(range(data.shape[0]))
        output = net(data)
        predictions = nd.argmax(output, axis=1)                
        predictions = predictions[remaining_idx]
        label = label[remaining_idx]
        acc.update(preds=predictions, labels=label)        
    return acc.get()[1]


def get_byz(byz_type):
    """
    简要概述: 根据字符串标识返回对应的拜占庭攻击函数。

    参数:
        byz_type (str): 攻击类型名称，如 'no'、'trim_attack'、'poisonedfl' 等。

    返回:
        callable: 满足统一接口的攻击函数。

    异常:
        NotImplementedError: 未支持的攻击类型。

    复杂度:
        时间 O(1); 空间 O(1)。

    费曼学习法:
        (A) 功能: 将文本配置翻译成具体的攻击函数实现。
        (B) 类比: 像根据菜单项取出对应的烹饪方法。
        (C) 步骤拆解:
            1. 依据输入字符串在 if-elif 序列中匹配。
            2. 返回 byzantine 模块中对应的函数引用。
            3. 若没有匹配项，抛出未实现异常提醒扩展。
        (D) 示例:
            - 调用: `attack_fn = get_byz('poisonedfl')`
            - 输出: `attack_fn` 可直接作为聚合前的攻击函数调用。
        (E) 边界与测试: 字符串必须准确匹配；可为每个分支写断言测试是否返回正确函数。
        (F) 背景与参考: 依赖 `byzantine` 模块的多种攻击策略实现。
    """
    if byz_type == "no":
        return byzantine.no_byz
    elif byz_type == 'trim_attack':
        return byzantine.fang_attack
    elif byz_type == 'lie_attack':
        return byzantine.lie_attack
    elif byz_type == 'dyn_attack':
        return byzantine.opt_fang
    elif byz_type == 'min_max':
        return byzantine.min_max
    elif byz_type == 'min_sum':
        return byzantine.min_sum
    elif byz_type == 'init_attack':
        return byzantine.init_attack
    elif byz_type == 'random_attack':
        return byzantine.random_attack
    elif byz_type == "poisonedfl":
        return byzantine.poisonedfl
    else:
        raise NotImplementedError
        
def load_data(dataset):
    """
    简要概述: 根据数据集名称载入训练与测试数据迭代器或缓存。

    参数:
        dataset (str): 数据集标识，如 'FashionMNIST'、'mnist'、'cifar10'、'purchase'、'FEMNIST'。

    返回:
        tuple: 训练数据与测试数据的迭代器或缓存结构，具体取决于数据集。

    异常:
        NotImplementedError: 当数据集名称不受支持。

    复杂度:
        时间 O(N); 空间 O(N)，取决于数据集加载实现。

    费曼学习法:
        (A) 功能: 根据配置准备相应数据源供联邦训练与评估使用。
        (B) 类比: 像依据菜单选择不同原材料的装配方式，有的直接烹饪，有的需预处理。
        (C) 步骤拆解:
            1. 判断数据集名称并设置对应的预处理函数。
            2. 调用 Gluon 数据集或自定义读取逻辑构建 DataLoader 或缓存。
            3. 返回训练与测试数据句柄，供后续客户端采样与评估。
        (D) 示例:
            - 调用: `train_data, test_data = load_data('FashionMNIST')`
            - 输出: `train_data`、`test_data` 均为 Gluon DataLoader。
        (E) 边界与测试: 需确保数据文件存在；针对 FEMNIST/purchase 等自定义数据要验证路径正确。
        (F) 背景与参考: 依赖 MXNet Gluon 数据接口与 LEAF/FEMNIST 数据格式，可参考对应数据集说明。
    """
    if dataset == 'FashionMNIST':
        def transform(data, label):
            return nd.transpose(data.astype(np.float32), (2, 0, 1)) / 255, label.astype(np.float32)
        # 使用 Gluon 内置数据集并标准化像素
        train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.FashionMNIST(train=True, transform=transform), 60000,shuffle=True, last_batch='rollover')
        test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.FashionMNIST(train=False, transform=transform), 250, shuffle=False, last_batch='rollover')
    elif dataset == 'mnist':
        def transform(data, label):
            return nd.transpose(data.astype(np.float32), (2, 0, 1)) / 255, label.astype(np.float32)
        # MNIST 与 FashionMNIST 预处理一致
        train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=True, transform=transform), 60000,shuffle=True, last_batch='rollover')
        test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=False, transform=transform), 250, shuffle=False, last_batch='rollover')
    elif dataset == 'cifar10':
        def transform(data, label):
            data = nd.transpose(data.astype(np.float32), (2, 0, 1)) / 255
            return data, label.astype(np.float32)
        # CIFAR10 使用彩色图像，批量大小按经验设定
        train_data = mx.gluon.data.DataLoader(
            mx.gluon.data.vision.CIFAR10(train=True, transform=transform),
            batch_size=128,
            shuffle=True,
            last_batch='rollover')
        test_data = mx.gluon.data.DataLoader(
            mx.gluon.data.vision.CIFAR10(train=False, transform=transform),
            batch_size=128,
            shuffle=False,
            last_batch='rollover')
    elif args.dataset == 'purchase':
        # 读取自建 purchase 数据集并按客户端划分缓存
        all_data = np.genfromtxt("./purchase/dataset_purchase", skip_header=1, delimiter=',')
        shuffle_index = np.random.permutation(all_data.shape[0])
        all_data = all_data[shuffle_index]
        each_worker_data = [nd.array(all_data[150*i:150*(i+1), 1:] * 2. - 1) for i in range(1200)]
        each_worker_label = [nd.array(all_data[150*i:150*(i+1), 0] - 1) for i in range(1200)]   
        train_data = (each_worker_data, each_worker_label)
        test_data = ((nd.array(all_data[180000:, 1:] * 2. - 1), nd.array(all_data[180000:, 0] - 1)),)
    elif args.dataset == "FEMNIST":
        # FEMNIST 使用 LEAF 格式，需按文件读取每个客户端数据
        each_worker_data = []
        each_worker_label = []
        each_worker_num = []
        for i in range(30):
            filestring = "./leaf/data/femnist/data/train/" + \
                "all_data_"+str(i) + "_niid_0_keep_100_train_9.json"
            with open(filestring, 'r') as f:
                load_dict = json.load(f)
                each_worker_num.extend(load_dict['num_samples'])
                for user in load_dict['users']:
                    x = nd.array(load_dict['user_data'][user]['x']).reshape(-1, 1, 28, 28)
                    y = nd.array(load_dict['user_data'] [user]['y'])

                    each_worker_data.append(x)
                    each_worker_label.append(y)

        # 随机打乱客户端顺序以平衡数据偏置
        random_order = np.random.RandomState(
            seed=args.seed).permutation(args.nworkers)
        each_worker_data = [each_worker_data[i] for i in random_order]
        each_worker_label = [each_worker_label[i] for i in random_order]
        each_worker_num = nd.array([each_worker_num[i]
                                   for i in random_order])
        train_data = (each_worker_data, each_worker_label)
        train_data_dir = os.path.join(
            "./leaf/data/femnist/data", "train")
        test_data_dir = os.path.join(
            "./leaf/data/femnist/data", "test")
        data = read_data(train_data_dir, test_data_dir)
        users, groups, train_data_ori, test_data_ori = data
        test_dataset = gluon.data.ArrayDataset(nd.concat(*[nd.array(test_data_ori[u]['x']).reshape(-1,1, 28, 28) for u in users], dim = 0), nd.concat(*[nd.array(test_data_ori[u]['y'])for u in users], dim = 0))
        test_data = gluon.data.DataLoader(test_dataset, batch_size=250, shuffle=False, last_batch='rollover')
    else: 
        raise NotImplementedError
    return train_data, test_data
    

def assign_data(train_data, bias, ctx, num_labels=10, num_workers=100, server_pc=100, p=0.1, dataset="FashionMNIST", seed=1, num_inputs=(1, 561)):
    """
    简要概述: 将全量数据划分给服务器与客户端，构造非独立同分布的数据划分。

    参数:
        train_data (Any): 训练数据集，结构因数据集而异。
        bias (float): 控制客户端数据偏斜程度的系数。
        ctx (mx.context.Context): MXNet 设备上下文。
        num_labels (int): 标签类别数，默认 10。
        num_workers (int): 客户端数量，默认 100。
        server_pc (int): 服务器持有样本数量，默认 100。
        p (float): 服务器采样标签偏置概率。
        dataset (str): 数据集名称。
        seed (int): 随机种子，保证划分可复现。
        num_inputs (tuple): 输入形状，用于构建数据张量。

    返回:
        tuple: (server_data, server_label, each_worker_data, each_worker_label, each_worker_num)，分别是服务器数据及每个客户端的数据与数量。

    异常:
        无。

    复杂度:
        时间 O(N); 空间 O(N)，N 为样本总数。

    费曼学习法:
        (A) 功能: 按设定的偏置规则把训练数据分配给服务器和客户端。
        (B) 类比: 像把一堆水果按种类和偏好分装进不同篮子，有的篮子偏爱某种水果。
        (C) 步骤拆解:
            1. 针对特殊数据集（如 purchase）直接按缓存结构挑选服务器样本。
            2. 对常规图像数据，先随机抽取服务器样本并根据偏置概率挑选标签。
            3. 构建每个客户端的索引池，按 bias 控制同类样本的集中度。
            4. 遍历客户端，将抽取的样本复制到设备上组成本地数据张量。
            5. 记录每个客户端拥有的样本数量供后续采样。
        (D) 示例:
            - 调用: `server_data, server_label, w_data, w_label, w_num = assign_data(train_data, 0.5, mx.cpu())`
            - 输出: 服务器与客户端数据划分完成，可直接用于本地训练。
        (E) 边界与测试: 需确保总体样本数 ≥ `num_workers`; 建议测试极端 bias=0 与 bias=1 的分布差异。
        (F) 背景与参考: 基于联邦学习常见的非 IID 划分设定，可参见《Measuring the Effects of Non-Identical Data Distribution》。
    """
    if dataset == "purchase":
        # purchase 数据集已经按客户端缓存，只需采样服务器份额
        server_data = []
        server_label = [] 
        for i in range(len(train_data[0])):
            if i >= server_pc:
                break
            # 随机挑选该客户端中的一个样本纳入服务器数据
            rd = random.randint(1, train_data[0][i].shape[0]-1)
            server_data.append(nd.expand_dims(train_data[0][i][rd], axis = 0))
            server_label.append(train_data[1][i][rd])
        server_data = nd.concat(*server_data, dim=0) if server_pc > 0 else None
        server_label = nd.concat(*server_label, dim=0) if server_pc > 0 else None
        return server_data, server_label, train_data[0], train_data[1]
    
    elif dataset == "FEMNIST":
        server_data = []
        server_label = [] 
        for i in range(len(train_data[0])):
            if i >= server_pc:
                break
            rd = random.randint(1, train_data[0][i].shape[0]-1)
            server_data.append(nd.expand_dims(train_data[0][i][rd], axis = 0))
            server_label.append(train_data[1][i][rd])
        server_data = nd.concat(*server_data, dim=0) if server_pc > 0 else None
        server_label = nd.concat(*server_label, dim=0) if server_pc > 0 else None
        return server_data, server_label, train_data[0], train_data[1]

    elif dataset == "FashionMNIST" or dataset == "mnist":
        # 将样本分配给各个客户端存储
        other_group_size = (1 - bias) / (num_labels - 1)
        worker_per_group = num_workers / num_labels

        #assign training data to each worker
        each_worker_data = [[] for _ in range(num_workers)]
        each_worker_label = [[] for _ in range(num_workers)]   
        server_data = []
        server_label = [] 
        
        # 计算服务器端各标签目标样本数
        real_dis = [1. / num_labels for _ in range(num_labels)]
        samp_dis = [0 for _ in range(num_labels)]
        num1 = int(server_pc * p)
        samp_dis[1] = num1
        average_num = (server_pc - num1) / (num_labels - 1)
        resid = average_num - np.floor(average_num)
        sum_res = 0.
        for other_num in range(num_labels - 1):
            if other_num == 1:
                continue
            samp_dis[other_num] = int(average_num)
            sum_res += resid
            if sum_res >= 1.0:
                samp_dis[other_num] += 1
                sum_res -= 1
        samp_dis[num_labels - 1] = server_pc - np.sum(samp_dis[:num_labels - 1])

        # 按标签概率分配样本到服务器与各客户端
        server_counter = [0 for _ in range(num_labels)]
        for _, (data, label) in enumerate(train_data):
            for (x, y) in zip(data, label):
                x = x.as_in_context(ctx).reshape(1,1,28,28)
                y = y.as_in_context(ctx)
                
                upper_bound = (y.asnumpy()) * (1. - bias) / (num_labels - 1) + bias
                lower_bound = (y.asnumpy()) * (1. - bias) / (num_labels - 1)
                rd = np.random.random_sample()
                
                if rd > upper_bound:
                    worker_group = int(np.floor((rd - upper_bound) / other_group_size) + y.asnumpy() + 1)
                elif rd < lower_bound:
                    worker_group = int(np.floor(rd / other_group_size))
                else:
                    worker_group = y.asnumpy()
                
                if server_counter[int(y.asnumpy())] < samp_dis[int(y.asnumpy())]:
                    server_data.append(x)
                    server_label.append(y)
                    server_counter[int(y.asnumpy())] += 1
                else:
                    rd = np.random.random_sample()
                    selected_worker = int(worker_group * worker_per_group + int(np.floor(rd * worker_per_group)))
                    each_worker_data[selected_worker].append(x)
                    each_worker_label[selected_worker].append(y)
                    
         
        server_data = nd.concat(*server_data, dim=0) if server_pc > 0 else None
        server_label = nd.concat(*server_label, dim=0) if server_pc > 0 else None
        
        each_worker_data = [nd.concat(*each_worker, dim=0) for each_worker in each_worker_data] 
        each_worker_label = [nd.concat(*each_worker, dim=0) for each_worker in each_worker_label]

        random_order = np.random.RandomState(seed=seed).permutation(num_workers)
        each_worker_data = [each_worker_data[i] for i in random_order]
        each_worker_label = [each_worker_label[i] for i in random_order]
        return server_data, server_label, each_worker_data, each_worker_label
    
    elif dataset == "cifar10":
        # 将样本分配给各个客户端存储
        other_group_size = (1 - bias) / (num_labels - 1)
        worker_per_group = num_workers / num_labels

        #assign training data to each worker
        each_worker_data = [[] for _ in range(num_workers)]
        each_worker_label = [[] for _ in range(num_workers)]
        server_data = []
        server_label = []

        # 计算服务器端各标签目标样本数
        real_dis = [1. / num_labels for _ in range(num_labels)]
        samp_dis = [0 for _ in range(num_labels)]
        num1 = int(server_pc * p)
        samp_dis[1] = num1
        average_num = (server_pc - num1) / (num_labels - 1)
        resid = average_num - np.floor(average_num)
        sum_res = 0.
        for other_num in range(num_labels - 1):
            if other_num == 1:
                continue
            samp_dis[other_num] = int(average_num)
            sum_res += resid
            if sum_res >= 1.0:
                samp_dis[other_num] += 1
                sum_res -= 1
        samp_dis[num_labels - 1] = server_pc - \
            np.sum(samp_dis[:num_labels - 1])

        # 按标签概率分配样本到服务器与各客户端
        server_counter = [0 for _ in range(num_labels)]
        for _, (data, label) in enumerate(train_data):
            for (x, y) in zip(data, label):
                x = x.as_in_context(ctx).reshape(1, 3, 32, 32)
                y = y.as_in_context(ctx)

                upper_bound = (y.asnumpy()) * (1. - bias) / \
                    (num_labels - 1) + bias
                lower_bound = (y.asnumpy()) * (1. - bias) / (num_labels - 1)
                rd = np.random.random_sample()

                if rd > upper_bound:
                    worker_group = int(
                        np.floor((rd - upper_bound) / other_group_size) + y.asnumpy() + 1)
                elif rd < lower_bound:
                    worker_group = int(np.floor(rd / other_group_size))
                else:
                    worker_group = y.asnumpy()

                if server_counter[int(y.asnumpy())] < samp_dis[int(y.asnumpy())]:
                    server_data.append(x)
                    server_label.append(y)
                    server_counter[int(y.asnumpy())] += 1
                else:
                    rd = np.random.random_sample()
                    selected_worker = int(
                        worker_group * worker_per_group + int(np.floor(rd * worker_per_group)))
                    each_worker_data[selected_worker].append(x)
                    each_worker_label[selected_worker].append(y)

        server_data = nd.concat(*server_data, dim=0) if server_pc > 0 else None
        server_label = nd.concat(
            *server_label, dim=0) if server_pc > 0 else None

        each_worker_data = [nd.concat(*each_worker, dim=0)
                            for each_worker in each_worker_data]
        each_worker_label = [nd.concat(*each_worker, dim=0)
                             for each_worker in each_worker_label]

        # 随机打乱客户端顺序以消除固定模式
        random_order = np.random.RandomState(
            seed=seed).permutation(num_workers)
        each_worker_data = [each_worker_data[i] for i in random_order]
        each_worker_label = [each_worker_label[i] for i in random_order]

        return server_data, server_label, each_worker_data, each_worker_label


def select_clients(clients, frac=1.0):
    """
    简要概述: 随机选择给定比例的客户端参与当前训练轮次。

    参数:
        clients (Iterable[int]): 全部客户端的索引集合。
        frac (float): 参与比例，取值范围 (0, 1]，默认 1.0 即全部参与。

    返回:
        list[int]: 被选中的客户端索引列表。

    异常:
        无。

    复杂度:
        时间 O(k); 空间 O(k)，其中 k 为客户端总数。

    费曼学习法:
        (A) 功能: 将总客户端名单按比例随机抽样，得到参与者列表。
        (B) 类比: 像从班级名单中抽取一定比例的学生参加实验。
        (C) 步骤拆解:
            1. 将输入集合转为列表，确保支持随机采样。
            2. 根据比例计算需要抽取的数量，并确保至少选中一人。
            3. 若比例小于 1，使用 `random.sample` 无放回抽样；否则返回原列表。
        (D) 示例:
            - 调用: `selected = select_clients(range(100), 0.1)`
            - 输出: `selected` 包含约 10 个唯一客户端索引。
        (E) 边界与测试: 当 `frac` 非法如 0 或 >1 时需预先校验；可测试 `frac=1` 是否返回原列表。
        (F) 背景与参考: 模拟联邦平均算法中的客户端随机参与机制。
    """
    client_list = list(clients)
    if frac < 1.0:
        # 抽样数量至少为 1，避免本轮无人参与
        num_clients = max(1, int(frac * len(client_list)))
        return random.sample(client_list, num_clients)
    return client_list
        

def main(args):
    """
    简要概述: 执行联邦学习主流程，包括数据准备、客户端训练、攻击注入与聚合。

    参数:
        args (argparse.Namespace): 由命令行解析得到的实验配置。

    返回:
        None: 函数以副作用形式完成训练与日志记录。

    异常:
        NotImplementedError: 当聚合器或攻击类型未实现时抛出。

    复杂度:
        时间 O(E·(C·L + A)); 空间 O(P)，其中 E 为全局轮数、C 为客户端数、L 为本地迭代次数、A 为聚合成本、P 为模型参数规模。

    费曼学习法:
        (A) 功能: 完整执行一次含拜占庭攻击的联邦训练实验并记录测试表现。
        (B) 类比: 像组织一场多轮比赛：先安排选手、准备赛道，再让部分选手接受“作弊”指令，最后统计比赛结果。
        (C) 步骤拆解:
            1. 根据配置确定运行设备、随机种子和数据形态。
            2. 加载数据并按非 IID 规则划分给服务器与客户端。
            3. 初始化模型、损失函数以及攻击与聚合模块。
            4. 在每一轮中随机挑选参与客户端，执行本地训练并收集梯度。
            5. 对恶意客户端应用攻击策略，再调用聚合器更新全局模型。
            6. 定期评估测试准确率并维护历史量以支持动态攻击。
        (D) 示例:
            - 调用: `main(parse_args())`
            - 输出: 在控制台打印各轮测试准确率。
        (E) 边界与测试: 需确保数据文件与 GPU 环境可用；可使用极小客户端数与轮数进行快速回归测试。
        (F) 背景与参考: 该流程综合了联邦学习、鲁棒聚合与拜占庭攻击研究中的常见组件，可参考 FedAvg 与相关攻击文献。
    """
    # 根据命令行参数决定运行设备
    ctx = get_device(args.gpu)

    # 配置基础超参数
    batch_size = args.batch_size
    num_inputs, num_outputs, num_labels = get_shapes(args.dataset)
    byz = get_byz(args.byz_type)
    num_workers = args.nworkers
    lr = args.lr
    niter = args.niter


    with ctx:
        net = get_net(args.net, num_outputs)

        net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), force_reinit=True, ctx=ctx)
        # 定义交叉熵损失作为分类目标
        softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

        grad_list = []
        test_acc_list = []

        # 加载原始训练与测试数据
        seed = args.seed
        if seed > 0:
            mx.random.seed(seed)
            random.seed(seed)
            np.random.seed(seed)
        train_data, test_data = load_data(args.dataset)
        
        # 按非独立同分布设定划分数据到服务器与客户端
        server_data, server_label, each_worker_data, each_worker_label = assign_data(
                                                                    train_data, args.bias, ctx, num_labels=num_labels, num_workers=num_workers, 
                                                                    server_pc=args.server_pc, p=args.p, dataset=args.dataset, seed=seed,num_inputs=num_inputs)
        # 进行一次前向传播，确保参数与缓存被真正初始化
        data_count = []
        for data in each_worker_data:
            data_count.append(data.shape[0])
        net(nd.zeros(num_inputs, ctx=ctx))

        # 记录初始模型及历史量，便于攻击与聚合逻辑推断
        init_model = [param.data().copy() for param in net.collect_params().values()]
        last_model = [param.data().copy() for param in net.collect_params().values()]
        history = None
        last_50_model = None
        last_grad = None
        sf = args.sf

        # 生成 PoisonedFL 攻击所需的固定随机方向
        fixed_rand = nd.sign(nd.random.normal(loc=0, scale=1, shape=nd.concat(
            *[xx.reshape((-1, 1)) for xx in init_model], dim=0).shape)).squeeze()

        avg_loss = 0

        # 开始全局训练循环        
        for e in range(niter):       
            participating_clients = select_clients(
                range(num_workers) , args.participation_rate)
            
            # 根据参与比例估计本轮恶意客户端数量
            probability = args.nfake * args.participation_rate - int(args.nfake * args.participation_rate)
            if random.random() >= probability:
                parti_nfake = int(args.nfake * args.participation_rate)
            else:
                parti_nfake = int(args.nfake * args.participation_rate) + 1
            
            # 初始化恶意客户端的梯度占位
            for i in range(parti_nfake):
                grad_list.append([nd.zeros_like(param.grad().copy()) for param in net.collect_params().values()])
                
            # 针对真实客户端执行本地训练并累计梯度
            for i in participating_clients:
                ori_para = [param.data().copy() for param in net.collect_params().values()]
                for _ in range(args.local_epoch):
                    shuffled_order = np.random.choice(list(range(each_worker_data[i].shape[0])), size=each_worker_data[i].shape[0], replace=False)
                    for b_id in range(max(each_worker_data[i].shape[0]//batch_size, 1)):
                        if batch_size >= each_worker_data[i].shape[0]:
                            minibatch = list(range(each_worker_data[i].shape[0]))
                        else:
                            minibatch = shuffled_order[b_id * batch_size: (b_id +1) * batch_size]
                        with autograd.record():
                            output = net(each_worker_data[i][minibatch])
                            loss = softmax_cross_entropy(
                                output, each_worker_label[i][minibatch])
                        loss.backward()
                        avg_loss += sum(loss)/len(loss)
                        for j, (param) in enumerate(net.collect_params().values()):
                            param.set_data(param.data().copy() - lr/batch_size * param.grad().copy())
                        
                grad_list.append([( param.data().copy()- ori_data.copy()) for param, ori_data in zip(net.collect_params().values(), ori_para)])
                for param, ori_data in zip(net.collect_params().values(), ori_para):
                    param.set_data(ori_data)
            try:
                avg_loss = (avg_loss/len(participating_clients)).asnumpy()[0]
            except:
                import pdb
                pdb.set_trace()

            avg_loss = 0
            if not grad_list:
                continue
            if args.aggregation == "mean":
                return_pare_list, sf = nd_aggregation.simple_mean(
                grad_list, net, lr / batch_size, parti_nfake, byz, history, fixed_rand, init_model, last_50_model, last_grad, sf, e)    
            elif args.aggregation == "trim":
                return_pare_list, sf = nd_aggregation.trim(
                    grad_list, net, lr / batch_size, parti_nfake, byz, history, fixed_rand, init_model, last_50_model, last_grad, sf, e)
            elif args.aggregation == "median":
                return_pare_list, sf = nd_aggregation.median(
                    grad_list, net, lr / batch_size, parti_nfake, byz, history, fixed_rand, init_model, last_50_model, last_grad, sf, e)
            elif args.aggregation == "mean_norm":
                return_pare_list, sf = nd_aggregation.mean_norm(
                    grad_list, net, lr / batch_size, parti_nfake, byz, history,fixed_rand, init_model, last_50_model, last_grad, sf, e)
            else:
                raise NotImplementedError
            if parti_nfake != 0:
                if "norm" in args.aggregation:
                    last_grad = nd.mean(return_pare_list[:,:parti_nfake], axis=-1).copy()
                else:
                    last_grad = nd.mean(
                        nd.concat(*return_pare_list[:parti_nfake], dim=1), axis=-1).copy()
            del grad_list
            del return_pare_list
            grad_list = []
            current_model = [param.data().copy() for param in net.collect_params().values()]
            if (e + 1) % args.step == 0 or e + 20 >= args.niter:
                test_accuracy = evaluate_accuracy(test_data, net, ctx)
                test_acc_list.append(test_accuracy)
                print("Iteration %02d. Test_acc %0.4f" % (e, test_accuracy))
                
            if e % 50 == 0:
                last_50_model = current_model
            history = (nd.concat(*[xx.reshape((-1, 1)) for xx in current_model], dim=0) - nd.concat(*[xx.reshape((-1, 1)) for xx in last_model], dim=0) )
            last_model = [param.data().copy() for param in net.collect_params().values()]
            

            from os import path
                
        del test_acc_list
        test_acc_list = []

   
if __name__ == "__main__":
    args = parse_args()
    main(args)


__AI_ANNOTATION_SUMMARY__ = """
parse_args: 构建并解析联邦学习实验的命令行参数。
get_device: 根据编号选择 MXNet 计算设备。
get_dnn: 构建两层全连接网络作为基础模型。
get_cnn: 构建适用于灰度图像的卷积神经网络。
get_cnn_cifar: 构建针对 CIFAR-10 的卷积网络结构。
get_net: 按类型名称返回对应的神经网络实例。
get_shapes: 返回数据集对应的输入与输出尺寸。
evaluate_accuracy: 在给定数据迭代器上评估模型准确率。
get_byz: 映射攻击类型标识到具体攻击函数。
load_data: 按数据集名称加载训练与测试数据。
assign_data: 根据偏置策略划分服务器与客户端数据。
select_clients: 随机抽样客户端参与一轮训练。
main: 执行包含拜占庭攻击的联邦训练主流程。
"""
