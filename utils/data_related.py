import random
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, RandomSampler, Sampler, Dataset

from utils import tools
from utils.datasets import DataSet, LazyDataSet
from utils.lazy_loader import LazyDataLoader


def single_argmax_accuracy(Y_HAT: torch.Tensor, Y: torch.Tensor) -> float:
    y_hat = torch.argmax(Y_HAT, dim=1)
    y = torch.argmax(Y, dim=1)
    cmp = (y_hat == y).type(Y.dtype)
    return float(sum(cmp))


def split_data(features, labels, train, test, valid=.0, shuffle=True) -> Tuple:
    """
    分割数据集为训练集、测试集、验证集（可选）
    :param shuffle: 是否打乱数据集
    :param labels: 标签集
    :param features: 特征集
    :param train: 训练集比例
    :param test: 测试集比例
    :param valid: 验证集比例
    :return: （训练特征集，训练标签集），（验证特征集，验证标签集），（测试特征集，测试标签集）
    """
    assert train + test + valid == 1.0, '训练集、测试集、验证集比例之和须为1'
    data_len = features.shape[0]
    train_len = int(data_len * train)
    valid_len = int(data_len * valid)
    test_len = data_len - train_len - valid_len
    # # 将高维特征数据打上id
    # features_ids = np.array([
    #     np.ones((1, *self.__features__.shape[2:])) * i
    #     for i in range(data_len)
    # ])
    # self.__features__ = np.concatenate((features_ids, self.__features__), 1)
    # 数据集打乱
    if shuffle:
        index = torch.randint(0, data_len, (data_len,))
        features = features[index]
        labels = labels[index]
    # 数据集分割
    print('splitting data...')
    # train_fea, valid_fea, test_fea = features.split((train_len, valid_len, test_len))
    # train_labels, valid_labels, test_labels = labels.split((train_len, valid_len, test_len))
    train_fea, valid_fea, test_fea = np.split(features, (train_len, train_len + valid_len))
    train_labels, valid_labels, test_labels = np.split(labels, (train_len, train_len + valid_len))
    return (train_fea, train_labels), (valid_fea, valid_labels), \
        (test_fea, test_labels)


def split_data_(dataset: DataSet | LazyDataSet, train=0.8, test=0.2, valid=.0, small_data: float = 1.,
                shuffle=True, ):
    """
    分割数据集为训练集、测试集、验证集（可选）
    :param shuffle: 是否打乱数据集
    :param labels: 标签集
    :param features: 特征集
    :param train: 训练集比例
    :param test: 测试集比例
    :param valid: 验证集比例
    :return: （训练特征集，训练标签集），（验证特征集，验证标签集），（测试特征集，测试标签集）
    """
    assert train + test + valid == 1.0, '训练集、测试集、验证集比例之和须为1'
    assert 0 < small_data <= 1, '所取数据集必须为源数据集非空子集！'
    # 数据集分割
    print('splitting data...')
    data_len = len(dataset)
    train_len = int(data_len * train)
    valid_len = int(data_len * valid)
    # 根据small_data随机取出部分数据集
    total_range = np.random.randint(0, data_len, int(small_data * data_len))
    train_range, valid_range, test_range = np.split(total_range, (train_len, train_len + valid_len))
    # # 将高维特征数据打上id
    # features_ids = np.array([
    #     np.ones((1, *self.__features__.shape[2:])) * i
    #     for i in range(data_len)
    # ])
    # self.__features__ = np.concatenate((features_ids, self.__features__), 1)
    ret = (r for r in (train_range, valid_range, test_range) if len(r) > 0)
    return [
        DataLoader(
            r, shuffle=True,
            # collate_fn=lambda d: int(d[0])  # 避免让数据升维。每次只抽取一个数字
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


def to_loader(dataset: DataSet | LazyDataSet, batch_size: int = None, sampler=None, shuffle=True,
              **kwargs) -> DataLoader or LazyDataLoader:
    """
    根据数据集类型转化为数据集加载器
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
    elif type(dataset) is DataSet:
        return DataLoader(
            dataset, batch_size, shuffle=shuffle, collate_fn=dataset.collate_fn, sampler=sampler, **kwargs
        )
