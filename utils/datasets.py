from abc import ABC
from typing import Iterable, Callable

import torch
from torch.utils.data import Dataset as torch_dataset, DataLoader


class DataSet(torch_dataset):
    def __init__(self, features, labels, collate_fn: Callable = None):
        """
        普通数据集，存储数据实际内容供DataLoader进行读取。
        :param features: 数据特征集
        :param labels: 数据标签集
        :param collate_fn: 数据预处理方法。DataLoader取出数据后，使用此方法对数据进行预处理。
        """
        assert isinstance(features, Iterable) and isinstance(labels, Iterable)
        assert len(features) == len(labels), f'特征集长度{len(features)}与标签集长度{len(labels)}不等！'
        self.__features__ = features
        self.__labels__ = labels
        self.collate_fn = collate_fn

    def __getitem__(self, item):
        return self.__features__[item], self.__labels__[item]

    def __len__(self):
        return len(self.__features__)

    def to(self, device: torch.device) -> None:
        self.__features__ = self.__features__.to(device)
        self.__labels__ = self.__labels__.to(device)

    # def to_loader(self, batch_size: int = None, shuffle=True, collate_fn=None, **kwargs) -> DataLoader:
    #     """
    #     将数据集转化为DataLoader
    #     # :param load_multiple: 懒加载单次加载的倍数。懒加载每次读取数据量规定为`load_multiple * batch_size`。仅在`lazy = True`时有效
    #     # :param read_fn: 懒加载数据加载器读取方法。仅在`lazy = True`时有效
    #     # :param lazy: 启用懒加载DataLoader
    #     :param collate_fn: 数据预处理函数
    #     :param batch_size: DataLoader每次供给的数据量。默认为整个数据集
    #     :param shuffle: 是否打乱
    #     :param kwargs: Dataloader额外参数
    #     :return: DataLoader对象
    #     """
    #     if not batch_size:
    #         batch_size = self.feature_shape[0]
    #     return DataLoader(
    #         self, batch_size, shuffle=shuffle, collate_fn=collate_fn, **kwargs
    #     )

    def get_subset(self, indices: Iterable):
        return DataSet(self[indices][0], self[indices][1])

    @property
    def feature_shape(self):
        return self.__features__.shape

    @property
    def label_shape(self):
        return self.__labels__.shape


class LazyDataSet(DataSet):
    def __init__(self, features, labels, load_multiple, read_fn, collate_fn=None):
        """
        懒加载数据集，只存储数据的索引供LazyDataLoader使用。
        LazyDataLoader取该数据集中实际的数据内容时，会使用`read_fn`方法进行数据内容的读取。
        :param features: 数据特征集
        :param labels: 数据标签集
        :param load_multiple: 懒加载单次加载的倍数。懒加载每次读取数据量规定为`load_multiple * batch_size`
        :param read_fn: 数据内容读取方法
        :param collate_fn: 数据预处理方法
        """
        self.load_multiple = load_multiple
        self.read_fn = read_fn
        super().__init__(features, labels, collate_fn)