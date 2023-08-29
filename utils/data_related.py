import random
from typing import Tuple, Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from utils import tools
from utils.datasets import DataSet, LazyDataSet


def single_argmax_accuracy(Y_HAT: torch.Tensor, Y: torch.Tensor) -> float:
    y_hat = torch.argmax(Y_HAT, dim=1)
    y = torch.argmax(Y, dim=1)
    cmp = (y_hat == y).type(Y.dtype)
    return float(sum(cmp))


# def split_data(features, labels, train, test, valid=.0, shuffle=True) -> Tuple:
#     """
#     分割数据集为训练集、测试集、验证集（可选）
#     :param shuffle: 是否打乱数据集
#     :param labels: 标签集
#     :param features: 特征集
#     :param train: 训练集比例
#     :param test: 测试集比例
#     :param valid: 验证集比例
#     :return: （训练特征集，训练标签集），（验证特征集，验证标签集），（测试特征集，测试标签集）
#     """
#     assert train + test + valid == 1.0, '训练集、测试集、验证集比例之和须为1'
#     data_len = features.shape[0]
#     train_len = int(data_len * train)
#     valid_len = int(data_len * valid)
#     test_len = data_len - train_len - valid_len
#     # # 将高维特征数据打上id
#     # features_ids = np.array([
#     #     np.ones((1, *self.__features__.shape[2:])) * i
#     #     for i in range(data_len)
#     # ])
#     # self.__features__ = np.concatenate((features_ids, self.__features__), 1)
#     # 数据集打乱
#     if shuffle:
#         index = torch.randint(0, data_len, (data_len,))
#         features = features[index]
#         labels = labels[index]
#     # 数据集分割
#     print('splitting data...')
#     # train_fea, valid_fea, test_fea = features.split((train_len, valid_len, test_len))
#     # train_labels, valid_labels, test_labels = labels.split((train_len, valid_len, test_len))
#     train_fea, valid_fea, test_fea = np.split(features, (train_len, train_len + valid_len))
#     train_labels, valid_labels, test_labels = np.split(labels, (train_len, train_len + valid_len))
#     return (train_fea, train_labels), (valid_fea, valid_labels), \
#         (test_fea, test_labels)


def split_data(dataset: DataSet | LazyDataSet, train=0.8, test=0.2, valid=.0):
    """
    分割数据集为训练集、测试集、验证集
    :param dataset: 分割数据集
    :param train: 训练集比例
    :param test: 测试集比例
    :param valid: 验证集比例
    :return: 各集合涉及下标DataLoader、只有比例>0的集合才会返回。
    """
    assert train + test + valid == 1.0, '训练集、测试集、验证集比例之和须为1'
    # 数据集分割
    print('splitting data...')
    data_len = len(dataset)
    train_len = int(data_len * train)
    valid_len = int(data_len * valid)
    train_range, valid_range, test_range = np.split(np.arange(data_len), (train_len, train_len + valid_len))
    ret = (r for r in (train_range, valid_range, test_range) if len(r) > 0)
    return [
        DataLoader(
            r, shuffle=True,
            collate_fn=lambda d: d[0]  # 避免让数据升维。每次只抽取一个数字
        )
        for r in ret
    ]


# def read_img(path: str, required_shape: Tuple[int, int] = None, mode: str = 'L') -> torch.Tensor:
def read_img(path: str, required_shape: Tuple[int, int] = None, mode: str = 'L',
             requires_id: bool = False) -> np.ndarray:
    """
    读取图片
    :param path: 图片所在路径
    :param required_shape: 需要将图片resize成的尺寸
    :param mode: 图片读取模式
    :param requires_id: 是否需要给图片打上ID
    :return: 图片对应numpy数组，形状为（通道，图片高，图片宽，……）
    """
    img_modes = ['L', 'RGB']
    assert mode in img_modes, f'不支持的图像模式{mode}！'
    img = Image.open(path).convert(mode)
    # 若有要求shape，则进行resize，边缘填充黑条
    if required_shape:
        img = tools.resize_img(img, required_shape)
    img = np.array(img)
    # 复原出通道。1表示样本数量维
    if mode == 'L':
        img_channels = 1
    elif mode == 'RGB':
        img_channels = 3
    img = img.reshape((img_channels, *img.shape[:2]))
    if requires_id:
        # 添加上读取文件名
        print(path.split('/')[-1])
        img = np.hstack((path.split('/')[-1], img))
    return img


class LazyDataLoader:
    def __init__(self, index_dataset: DataSet, read_fn, batch_size: int = None, load_multiple: int = 1,
                 shuffle=True, collate_fn=None, sampler=None,
                 **kwargs):
        self.__batch_size = batch_size
        self.__multiple = load_multiple
        self.__shuffle = shuffle
        self.__collate_fn = collate_fn
        self.__read_fn = read_fn
        self.__sampler = sampler
        self.__kwargs = kwargs

        # self.__index_loader__ = index_dataset.to_loader(batch_size * load_multiple, shuffle)
        self.__index_loader = to_loader(index_dataset, batch_size * load_multiple, shuffle=shuffle)
        pass

    def __iter__(self):
        for index, label in self.__index_loader:
            # batch_loader = DataSet(self.__read_fn(index), label).to_loader(
            #     self.__batch_size, self.__shuffle, self.__collate_fn, **self.__kwargs
            # )
            batch_loader = to_loader(
                DataSet(self.__read_fn(index), label),
                self.__batch_size, self.__sampler, self.__shuffle, **self.__kwargs
            )
            for X, y in batch_loader:
                yield X, y

    def __len__(self):
        return len(self.__index_loader) * self.__multiple


def to_loader(dataset: DataSet | LazyDataSet, batch_size: int = None, sampler: Iterable = None, shuffle=True,
              **kwargs) -> DataLoader or LazyDataLoader:
    """
    根据数据集类型转化为数据集加载器
    :param sampler: 实现了__len__()的可迭代对象，用于供给下标。若不指定，则使用默认sampler.
    :param dataset: 转化为加载器的数据集。
    :param batch_size: 每次供给的数据量。默认为整个数据集
    :param shuffle: 是否打乱
    :param kwargs: Dataloader额外参数
    :return: 加载器对象
    """
    if sampler is not None:
        shuffle = None
    if not batch_size:
        batch_size = dataset.feature_shape[0]
    if type(dataset) == LazyDataSet:
        return LazyDataLoader(
            dataset, dataset.read_fn, batch_size, load_multiple=dataset.load_multiple, shuffle=shuffle,
            collate_fn=dataset.collate_fn, sampler=sampler, **kwargs
        )
    elif type(dataset) == DataSet:
        return DataLoader(
            dataset, batch_size, shuffle=shuffle, collate_fn=dataset.collate_fn, sampler=sampler, **kwargs
        )


def k_fold_split(dataset: DataSet or LazyDataSet, k: int = 10, shuffle: bool = True):
    """
    根据K、i、X、y获取训练集和验证集
    :param shuffle:
    :param dataset:
    :param k: 数据集拆分折数
    :param i:
    :param X:
    :param y:
    :return:
    """
    assert k > 1, f'k折验证需要k值大于1，而不是{k}'
    # assert 0 <= i < k, f'i值需要介于[0, k)间，且为整数，输入的i值为{i}'
    # fold_size = len(dataset) // k
    # X_train, y_train = None, None
    # for j in range(k):
    #     idx = slice(j * fold_size, (j + 1) * fold_size)
    #     X_part, y_part = X[idx, :], y[idx]
    #     if j == i:
    #         X_valid, y_valid = X_part, y_part
    #     elif X_train is None:
    #         X_train, y_train = X_part, y_part
    #     else:
    #         X_train = torch.cat([X_train, X_part], 0)
    #         y_train = torch.cat([y_train, y_part], 0)
    # return X_train, y_train, X_valid, y_valid
    data_len = len(dataset)
    fold_size = len(dataset) // k
    total_ranger = np.random.randint(0, data_len, (data_len, )) if shuffle else np.arange(data_len)
    for i in range(k):
        train_range1, valid_range, train_range2 = np.split(
            total_ranger,
            (i * fold_size, min((i + 1) * fold_size, data_len))
        )
        train_range = np.concatenate((train_range1, train_range2), axis=0)
        del train_range1, train_range2
        yield [
            DataLoader(ranger, shuffle=True, collate_fn=lambda d: d[0])
            for ranger in (train_range, valid_range)
        ]


def data_slicer(data, data_portion=1., shuffle=True) -> np.ndarray:
    assert 0 <= data_portion <= 1.0, '切分的数据集需为源数据集的子集！'
    if isinstance(data, np.ndarray):
        shuffle_fn = np.random.shuffle
    else:
        shuffle_fn = random.shuffle
    if shuffle:
        shuffle_fn(data)
    return data[:int(data_portion * len(data))]
