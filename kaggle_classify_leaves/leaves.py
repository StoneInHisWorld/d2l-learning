import os
import warnings
from typing import Tuple, Iterable, Callable

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from utils.data_related import read_img, data_slicer
from utils.datasets import DataSet, LazyDataSet


class LeavesTrain:
    img_channels = 3

    def __init__(self, where: str, lazy=True, required_shape: Tuple[int, int] = None, small_data=1.,
                 device: torch.device = 'cpu'):
        """
        按顺序保存全部树叶数据。
        :param where: 全部数据存放的目录，即train.csv等文件的所属根目录
        :param required_shape: 是否需要重塑读取图片的形状，None则为不需要，否则将图片重塑为指定元组形状
        :param device: 将数据迁移到所属设备上
        :param lazy: 是否进行懒加载
        """
        self.__where = where
        self.__required_shape = required_shape
        self.__device = device
        self.__lazy = lazy

        self.__check_path(where)
        assert 0 < small_data <= 1., '所取数据集需为源数据集的非空子集'
        if lazy:
            warnings.warn('将要进行懒加载。虽然这会降低内存消耗，但会极大地影响训练性能！')
            self.__read_data_lazily(small_data)
        else:
            self.__read_data(required_shape, small_data)
        assert len(self.__features) == len(self.__labels), '特征集和标签集长度须一致'

    def __read_data_lazily(self, small_data):
        print('reading indexes...')
        train_data = pd.read_csv(self.__train_dir__).values
        if small_data < 1:
            train_data, = data_slicer(small_data, True, train_data)
        # 提取测试文件中的信息
        img_paths, labels = [], []
        for path, label in train_data:
            img_paths.append(os.path.join(self.__where, path))
            labels.append(label)
        # 将标签转化为独热编码
        print('getting dummies...')
        labels = pd.get_dummies(labels)
        self.__dummy_columns__ = labels.columns
        self.__labels = labels.values
        self.__features = np.array(img_paths)

    def __read_data(self, required_shape, small_data):
        train_features, train_labels = [], []
        train_data = pd.read_csv(self.__train_dir__).values
        if small_data < 1:
            train_data, = data_slicer(small_data, True, train_data)
        # 取出文件中的数据
        with tqdm(train_data, desc='reading data...', unit='img') as pbar:
            for img_path, label in pbar:
                feature = read_img(os.path.join(self.__where, img_path), mode='RGB',
                                   required_shape=required_shape)
                train_features.append(feature)
                train_labels.append(label)
        print('turing data into tensor...')
        self.__features = torch.from_numpy(np.array(train_features))
        # 获取独热编码并转化为张量
        self.__labels = pd.get_dummies(train_labels)
        self.__dummy_columns__ = self.__labels.columns
        self.__labels = torch.tensor(self.__labels.values)
        print(f'typing data, and moving to {self.device}')
        # 类型转移
        self.__features = self.__features.to(torch.float32)
        self.__labels = self.__labels.to(torch.float32)
        # 设备转移
        self.__features = self.__features.to(self.device)
        self.__labels = self.__labels.to(self.device)

    def __check_path(self, path: str):
        path_iter = os.walk(path)
        _, folders, files = next(path_iter)
        if 'images' in folders:
            self.__images_dir__ = os.path.join(path, 'images')
        else:
            raise FileNotFoundError('该目录下无\"images\"文件夹！')
        if 'train.csv' in files:
            self.__train_dir__ = os.path.join(path, 'train.csv')
        else:
            raise FileNotFoundError('该目录下无\"train.csv\"文件！')

    def to_dataset(self, load_multiple: float = 1.):
        """
        根据自身模式，转换为合适的数据集
        :param load_multiple: 若为懒加载，则可指定本参数，决定一次性加载多少数据
        :return: pytorch框架下数据集
        """
        if self.__lazy:
            return LazyDataSet(self.__features, self.__labels, load_multiple, read_fn=self.read_fn)
        return DataSet(self.__features, self.__labels)

    def read_fn(self, index):
        X = []
        for path in index:
            X.append(read_img(path, required_shape=self.__required_shape, mode='RGB'))
        return np.array(X)

    @property
    def dummy(self) -> pd.Index:
        return self.__dummy_columns__

    @property
    def device(self):
        return self.__device

    @staticmethod
    def collate_fn(data: list):
        return data

    @staticmethod
    def features_preprocess(data):
        mean = data.mean(dim=0, keepdim=True)
        std = data.std(dim=0, keepdim=True)
        numerator = data - mean
        return numerator / std


class LeavesTest:

    def __init__(self, where: str, required_shape: Tuple[int, int] = None, device: torch.device = 'cpu'):
        """
        树叶测试集，存储所有树叶索引。树叶图片将在调用时，通过collate_fn()读取。
        :param where: 读取目标目录，为test.csv对应路径
        :param required_shape: 要求图片转化的形状。该形状作用于collate_fn输出的图片，便于网络训练
        :param device: 数据集所处设备
        """
        self.__check_path(where)
        self.__where = where
        self.__required_shape = required_shape
        self.device = device

        print('reading test_data...')
        self.__read_data()

    def __read_data(self):
        print('reading indexes...')
        self.__features = pd.read_csv(self.__test_dir)

    def __check_path(self, path: str):
        path_iter = os.walk(path)
        _, folders, files = next(path_iter)
        if 'images' in folders:
            self.__images_dir__ = os.path.join(path, 'images')
        else:
            raise FileNotFoundError('该目录下无\"images\"文件夹！')
        if 'test.csv' in files:
            self.__test_dir = os.path.join(path, 'test.csv')
        else:
            raise FileNotFoundError('该目录下无\"test.csv\"文件！')

    def collate_fn(self, data: Iterable):
        X = []
        for path in data:
            X.append(read_img(
                os.path.join(self.__where, path[0]), required_shape=self.__required_shape, mode='RGB'
            ))
        return torch.from_numpy(np.array(X)).to(torch.float32).to(self.device)

    # def apply(self, features_calls: list[Callable[[torch.Tensor], torch.Tensor]] = None):
    #     if features_calls is None:
    #         features_calls = []
    #     for call in features_calls:
    #         self.__features = call(self.__features)

    def to(self, device: torch.device = 'cpu'):
        self.device = device

    def __getitem__(self, item):
        return self.__features.iloc[item]

    @property
    def img_paths(self):
        return self.__features

    @property
    def imgs(self):
        return self.collate_fn(self.__features.values)
