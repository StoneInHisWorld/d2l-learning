import numpy as np
import torch
from PIL import Image
from typing import Tuple, Callable

from torch.utils.data import DataLoader

from utils import tools
from utils.datasets import DataSet
from utils.lazy_loader import LazyDataLoader


def single_argmax_accuracy(Y_HAT: torch.Tensor, Y: torch.Tensor) -> float:
    y_hat = torch.argmax(Y_HAT, dim=1)
    y = torch.argmax(Y, dim=1)
    cmp = (y_hat == y).type(Y.dtype)
    return float(sum(cmp))


def split_data(features, labels, train, test, valid=.0,
               shuffle=True):
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


# def read_img(path: str, required_shape: Tuple[int, int] = None, mode: str = 'L') -> torch.Tensor:
def read_img(path: str, required_shape: Tuple[int, int] = None, mode: str = 'L') -> np.ndarray:
    img_modes = ['L', 'RGB']
    assert mode in img_modes, f'不支持的图像模式{mode}！'
    img = Image.open(path).convert(mode)
    # 若有要求shape，则进行resize，边缘填充黑条
    if required_shape:
        img = tools.resize_img(img, required_shape)
    img = np.array(img)
    # img = torch.as_tensor(np.array(img), dtype=torch.float32)
    # 复原出通道。1表示样本数量维
    if mode == 'L':
        img_channels = 1
    elif mode == 'RGB':
        img_channels = 3
    img = img.reshape((img_channels, *img.shape[:2]))
    # print(path.split('/')[-1])
    # img = torch.hstack((path.split('/')[-1], img))
    return img


def to_loader(dataset: DataSet, batch_size: int = None, shuffle=True, collate_fn=None, lazy: bool = False,
              read_fn: Callable = None, load_multiple: int = 1,
              **kwargs) -> DataLoader or LazyDataLoader:
    """
    将数据集转化为DataLoader
    :param dataset:
    :param load_multiple: 懒加载单次加载的倍数。懒加载每次读取数据量规定为`load_multiple * batch_size`。仅在`lazy = True`时有效
    :param read_fn: 懒加载数据加载器读取方法。仅在`lazy = True`时有效
    :param lazy: 启用懒加载DataLoader
    :param collate_fn: 数据预处理函数
    :param batch_size: DataLoader每次供给的数据量。默认为整个数据集
    :param shuffle: 是否打乱
    :param kwargs: Dataloader额外参数
    :return: DataLoader对象
    """
    if not batch_size:
        batch_size = dataset.feature_shape[0]
    if lazy:
        return LazyDataLoader(
            dataset, read_fn, batch_size, load_multiple=load_multiple, shuffle=shuffle,
            collate_fn=collate_fn, **kwargs
        )
    else:
        return DataLoader(
            dataset, batch_size, shuffle=shuffle, collate_fn=collate_fn, **kwargs
        )
