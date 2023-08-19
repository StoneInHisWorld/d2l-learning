import os
from typing import Iterable, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset as torch_dataset, DataLoader

from utils.data_related import read_img, split_data


class DataSet(torch_dataset):
    def __init__(self, features, labels):
        assert isinstance(features, Iterable) and isinstance(labels, Iterable)
        assert len(features) == len(labels), f'特征集长度{len(features)}与标签集长度{len(labels)}不等！'
        self.__features__ = features
        self.__labels__ = labels

    def __getitem__(self, item):
        return self.__features__[item], self.__labels__[item]

    def __len__(self):
        return len(self.__features__)

    def to(self, device: torch.device) -> None:
        self.__features__ = self.__features__.to(device)
        self.__labels__ = self.__labels__.to(device)

    def to_loader(self, batch_size: int = None, shuffle=True, collate_fn=None,
                  **kwargs) -> DataLoader:
        """
        将数据集转化为DataLoader
        :param collate_fn:
        :param batch_size: DataLoader每次供给的数据量。默认为整个数据集
        :param shuffle: 是否打乱
        :param kwargs: Dataloader额外参数
        :return: DataLoader对象
        """
        if not batch_size:
            batch_size = self.feature_shape[0]
        return DataLoader(self, batch_size, shuffle=shuffle, collate_fn=collate_fn,
                          **kwargs)

    def get_subset(self, indices: Iterable):
        return DataSet(self[indices][0], self[indices][1])

    @property
    def feature_shape(self):
        return self.__features__.shape

    @property
    def label_shape(self):
        return self.__labels__.shape

