import hashlib
import os
import tarfile
import zipfile
from functools import reduce

import numpy as np
import pandas as pd
import requests
import torch
from d2l import torch as d2l
from torch import nn
from tqdm import tqdm

print(torch.cuda.is_available())


def download(name, cache_dir=os.path.join('..', 'data')):  # @save
    """下载一个DATA_HUB中的文件，返回本地文件名"""
    assert name in DATA_HUB, f"{name} 不存在于 {DATA_HUB}"
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname  # 命中缓存
    print(f'正在从{url}下载{fname}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname


def download_extract(name, folder=None):  # @save
    """下载并解压zip/tar文件"""
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, '只有zip/tar文件可以被解压缩'
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir


def download_all():  # @save
    """下载DATA_HUB中的所有文件"""
    for name in DATA_HUB:
        download(name)


def to_netDevice(net, *toTransfer):
    """
    将所需数据迁移到神经网络所在设备。
    :issue: 在函数以及for循环中对数据的迁移无效！
    :param net: 将要用于计算的神经网络
    :param toTransfer: 所有需要迁移的变量
    :return: 转换完成的数据变量
    """
    state = net.state_dict()
    devices = set()
    # 获取神经网络所有层所在设备
    for v in state.values():
        devices.add(v.device)
    num_device = len(devices)
    for device in devices:
        devices = device
        break
    # 若神经网络中多层变量位于不同设备上
    if num_device > 1:
        # 将所有层迁移到第一层所在设备上
        for v in state.values():
            v.to(devices)
    # 将需要转移的变量放到神经网络的设备上
    for t in toTransfer:
        if t is torch.Tensor:
            t.to(devices)
    return toTransfer


def try_gpu(i=0):
    """获取一个GPU"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def get_net():
    net = nn.Sequential(nn.Linear(in_features, 1))
    net.to(try_gpu())
    # print(net)
    return net


def get_net(num_in, num_out, dropout_rate=0, activation=nn.ReLU, type='single'):
    """
    获取一个神经网络
    :param num_in: 输入维度
    :param num_out: 输出维度
    :param dropout_rate: dropout比例
    :param activation: 激活函数
    :param type: 网络类型，可选“multi”，"self-defined“，默认为“single”
    :return: 神经网络对象（nn.Module），若有GPU则会迁移至GPU
    """
    if type == 'multi':
        """多层神经网络，逐层递减大小"""
        layers = []
        layer_sizes = np.logspace(np.log2(num_in), np.log2(num_out), int(np.log2(num_in - num_out)), base=2)
        layer_sizes = list(map(int, layer_sizes))
        for i in range(len(layer_sizes) - 1):
            in_size, out_size = layer_sizes[i], layer_sizes[i + 1]
            layers.append(nn.Linear(in_size, out_size))
            layers.append(activation())
        net = nn.Sequential(*layers)
    elif type == 'self-defined':
        """自定义网络大小"""
        net = nn.Sequential(
            nn.Linear(num_in, 128), activation(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, 2), activation(),
            nn.Linear(2, num_out)
        )
    else:
        """获取单层网络"""
        net = nn.Sequential(nn.Linear(in_features, 1))

    net.to(try_gpu())
    return net


def log_rmse(net, features, labels):
    # 为了在取对数时进一步稳定该值，将小于1的值设置为1
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds),
                           torch.log(labels)))
    return rmse.item()


def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    # 数据迁移
    # to_netDevice(net, train_features, train_labels, test_features, test_labels)
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    # 这里使用的是Adam优化算法
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    pbar = tqdm(range(num_epochs))
    for _ in pbar:
        pbar.set_description(f'Epoch{_}')
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    pbar.close()
    return train_ls, test_ls


def get_k_fold_data(k, i, X, y):
    """
    根据K、i、X、y获取训练集和验证集
    :param k:
    :param i:
    :param X:
    :param y:
    :return:
    """
    assert k > 1, 'k折交叉验证需要k值大于1！'
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid


def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay,
           batch_size, dropout_rate, activation, netType):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        # net = get_net()
        net = get_net(
            X_train.shape[1], y_train.shape[1],
            dropout_rate=dropout_rate, activation=activation, type=netType
        )
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        print(f"\tfold{i} done")
    return train_l_sum / k, valid_l_sum / k


def train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size, dropout_rate,
                   activation, netType):
    net = get_net(
        train_features.shape[1], train_labels.shape[1],
        dropout_rate=dropout_rate, activation=activation, type=netType
    )
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    print(f'训练log rmse：{float(train_ls[-1]):f}')
    # 将网络应用于测试集。
    preds = net(test_features).detach().numpy()
    # 将其重新格式化以导出到Kaggle
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    # test_data['Sold Price'] = pd.Series(preds.reshape(1, -1)[0])
    # submission = pd.concat([test_data['Id'], test_data['Sold Price']], axis=1)
    submission.to_csv('submission.csv', index=False)


# @save
DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'
# 加载数据集
DATA_HUB['kaggle_house_train'] = (  # @save
    DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce')
DATA_HUB['kaggle_house_test'] = (  # @save
    DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')
train_data = pd.read_csv(download('kaggle_house_train'))
test_data = pd.read_csv(download('kaggle_house_test'))
# train_data = pd.read_csv('train.csv')
# test_data = pd.read_csv('test.csv')

"""数据预处理"""
# 去掉ID列
all_features = pd.concat((train_data.iloc[:, 1:], test_data.iloc[:, 1:]), sort=False)
# 若无法获得测试数据，则可根据训练数据计算均值和标准差
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std())
)  # 对所有特征进行标准化
# 在标准化数据之后，所有均值消失，因此我们可以将缺失值设置为0
all_features[numeric_features] = all_features[numeric_features].fillna(0)

"""创建独热编码"""
no_dummy_column = [
    'Address', 'Summary'
    # , 'Bedrooms'
    # , 'Parking', 'Cooling', 'Heating'
    # , 'Elementary School', 'Middle School',
    # 'High School', 'Listed On'
]
no_dummy_column += [x for x in numeric_features.values]
"""多线程创建独热编码"""
# “Dummy_na=True”将“na”（缺失值）视为有效的特征值，并为其创建指示符特征
# import threading
#
#
# def get_onehotCodes(data, thread_id):
#     print(f'thread{thread_id} is working...')
#     # print(all_features.dtypes[all_features.dtypes != 'object'].index.values)
#     no_dummy_column = [
#         'Address', 'Summary'
#         # , 'Bedrooms'
#         # , 'Parking', 'Cooling', 'Heating'
#         # , 'Elementary School', 'Middle School',
#         # 'High School', 'Listed On'
#     ]
#     no_dummy_column += [x for x in numeric_features.values]
#     dummy_batch = pd.get_dummies(
#         data, dummy_na=True, dtype=np.dtype('int8'), columns=[x for x in data.columns if x not in no_dummy_column], sparse=True
#     )
#     features_oneHotCodes[thread_id] = dummy_batch
#     del dummy_batch
#
# import time
#
#
# num_thread = 16
# features_oneHotCodes = {}
# i = 0
# start = time.time()
# for i in range(num_thread):
#     batch_size = int(len(all_features) / num_thread)
#     tail_index = (i + 1) * batch_size if i is not num_thread - 1 else -1
#     feature_batch = all_features.iloc[i * batch_size: tail_index]
#     t = threading.Thread(target=get_onehotCodes, args=[feature_batch, i])
#     t.run()
#     del feature_batch
# end = time.time()
# print(end - start)
#
# all_features = pd.DataFrame()
#
# for thread_id in range(num_thread):
#     print(f"thread{thread_id - 1} is been concatenating to thread{thread_id}")
#     all_features = pd.concat(
#         [all_features, features_oneHotCodes[thread_id]], sort=False, axis=1
#     )
#     del features_oneHotCodes[thread_id]
#
"""防止数据集太大，单线程小批量转换独热编码"""
# batch = 100
# i = 0
# while i < len(all_features) and i + batch < len(all_features):
#     print(i)
#     dummies_batch = pd.get_dummies(all_features.iloc[i:i+batch, :], dummy_na=True, dtype=bool)
#     features_oneHotCodes = pd.concat(
#         [features_oneHotCodes, dummies_batch], sort=False, axis=0
#     )
#     i += batch
# if i < len(all_features):
#     features_oneHotCodes = pd.concat(
#         [features_oneHotCodes, all_features.iloc[i:-1, :]], sort=False
#     )

"""单线程一次性生成独热编码"""
all_features = pd.get_dummies(
    all_features, dummy_na=True, dtype=np.dtype('int8'), sparse=True,
    columns=[x for x in all_features.columns if x not in no_dummy_column]
)

n_train = train_data.shape[0]
# print(all_features.shape)
# print(all_features.dtypes)
all_features = all_features.fillna(0)
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
train_labels = torch.tensor(
    train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32
)

"""训练"""
loss = nn.MSELoss()
in_features = train_features.shape[1]


def get_parameters():
    """调参部分"""
    # 最合适值：k:6， num_epochs:325，lr:7，weight_decay:0.001，batch_size:16
    k_toUse = [3]
    numEpochs_toUse = [50]
    lr_toUse = [0.001]
    weightDecay_toUse = [0.1]
    batchSize_toUse = [16]
    # dropOutRate_toUse = np.linspace(0, 1, 10).tolist()
    dropOutRate_toUse = [0.1]
    # activations_toUse = [nn.Sigmoid, nn.ReLU, nn.Tanh, nn.LeakyReLU]
    activations_toUse = [nn.LeakyReLU]
    # netType_toUse = ["multi", "self-defined", "single"]
    netType_toUse = ["multi"]
    parameters_lists = []
    for k in k_toUse:
        for num_epoch in numEpochs_toUse:
            for lr in lr_toUse:
                for weight_decay in weightDecay_toUse:
                    for batch_size in batchSize_toUse:
                        for dropout_rate in dropOutRate_toUse:
                            for activation in activations_toUse:
                                for netType in netType_toUse:
                                    # yield [k, num_epoch, lr, weight_decay, batch_size, dropout_rate, activation,
                                    #        netType]
                                    parameters_lists += [
                                        [k, num_epoch, lr, weight_decay, batch_size, dropout_rate, activation, netType]
                                    ]
    return parameters_lists


optimal_parameters = []
min_ls = float('inf')
reduce()
for parameters in get_parameters():
    k, num_epochs, lr, weight_decay, batch_size, dropout_rate, activation, netType = parameters
    # pbar.set_description(f'超参数：1、k-折：{k}，2、迭代周期：{num_epochs}，3、学习率：{lr}，4、权重衰退：{weight_decay}，\n'
    #       f'5、批量大小：{batch_size}，6、dropout比例：{dropout_rate}，7、激活函数：{activation}，8、网络类型{netType}\n')
    print(f'超参数：1、k-折：{k}，2、迭代周期：{num_epochs}，3、学习率：{lr}，4、权重衰退：{weight_decay}，'
          f'5、批量大小：{batch_size}，6、dropout比例：{dropout_rate}，7、激活函数：{activation}，8、网络类型{netType}')
    train_l, valid_l = k_fold(
        k, train_features, train_labels, num_epochs, lr,
        weight_decay, batch_size, dropout_rate, activation, netType
    )
    print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, '
          f'平均验证log rmse: {float(valid_l):f}')
    if min_ls > valid_l:
        optimal_parameters = parameters
        min_ls = valid_l

print("选中的参数为", optimal_parameters)
k, num_epochs, lr, weight_decay, batch_size, dropout_rate, activation, netType = optimal_parameters
train_and_pred(
    train_features, test_features, train_labels, test_data,
    num_epochs, lr, weight_decay, batch_size, dropout_rate, activation, netType
)
