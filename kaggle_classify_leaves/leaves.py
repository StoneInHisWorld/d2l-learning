import os
import warnings
from typing import Tuple, Iterable

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from utils.data_related import read_img
from utils.datasets import DataSet, LazyDataSet
from utils.tools import data_slicer


# class LeavesTrain:
#     img_channels = 3
#
#     def __init__(self, where: str, train=0.8, valid=0.2, small_data=1., shuffle=True,
#                  lazy=True,
#                  required_shape: Tuple[int, int] = None, device: torch.device = 'cpu'):
#         """
#         使用懒加载模式，只读取数据索引并存储，数据本身的加载留给DataLoader
#         :param where: 全部数据存放的目录，即train.csv等文件的所属根目录
#         :param train: 训练集比例
#         :param valid: 验证集比例
#         :param small_data: 选取数据的比例
#         :param shuffle: 是否打乱数据集
#         :param required_shape: 是否需要重塑读取图片的形状，None则为不需要，否则将图片重塑为指定元组形状
#         :param device: 将数据迁移到所属设备上
#         :param lazy: 是否进行懒加载
#         """
#         self.__where__ = where
#         self.__required_shape__ = required_shape
#         self.__device__ = device
#         self.__lazy__ = lazy
#
#         self.__check_path(where)
#         if lazy:
#             warnings.warn('将要进行懒加载。虽然这会降低内存消耗，但会极大地影响训练性能！')
#             self.__read_data_lazily(small_data, shuffle)
#         else:
#             self.__read_data(required_shape, small_data, shuffle)
#         # # 按照获取比例抛弃部分数据集，打乱顺序后切换设备
#         # data_len = len(self.__features__)
#         # indices = torch.randint(0, data_len, (data_len, ))[:int(small_data * data_len)]
#         # self.__features__ = self.__features__[indices]
#         # self.__labels__ = self.__labels__[indices]
#         # # self.__features__ = self.__features__.to(device)
#         # # self.__labels__ = self.__labels__.to(device)
#         assert len(self.__features) == len(self.__labels), '特征集和标签集长度须一致'
#         # 打乱数据后，切分训练集、测试集、验证集
#         self.__train_data__, _, self.__valid_data__ = split_data(
#             self.__features, self.__labels, train, valid, shuffle=False
#         )
#
#     def __read_data_lazily(self, small_data, shuffle=True):
#         train_data = pd.read_csv(self.__train_dir__)
#         # 提取测试文件中的信息
#         img_paths = []
#         for path, _ in train_data.values:
#             img_paths.append(os.path.join(self.__where__, path))
#         features_paths, labels = img_paths, train_data['label']
#         # 将标签转化为独热编码
#         print('getting dummies...')
#         labels = pd.get_dummies(labels.values)
#         self.__dummy_columns__ = labels.columns
#         self.__labels = labels.values
#         self.__features = np.array(img_paths)
#         # 取出部分数据集并打乱
#         if shuffle:
#             # 按照获取比例抛弃部分数据集，打乱顺序后切换设备
#             data_len = len(self.__features)
#             indices = torch.randint(0, data_len, (data_len,))[:int(small_data * data_len)]
#             self.__features = self.__features[indices]
#             self.__labels = self.__labels[indices]
#
#     def __read_data(self, required_shape, small_data, shuffle=True):
#         train_features, train_labels = [], []
#         train_data = pd.read_csv(self.__train_dir__)
#         # 取出文件中的数据
#         with tqdm(train_data.values, desc='reading data...', unit='img') as pbar:
#             for img_path, label in pbar:
#                 feature = read_img(os.path.join(self.__where__, img_path), mode='RGB',
#                                    required_shape=required_shape)
#                 train_features.append(feature)
#                 train_labels.append(label)
#         print('turing data into tensor...')
#         self.__features = torch.from_numpy(np.array(train_features)).to(torch.float32)
#         # 获取独热编码并转化为张量
#         self.__labels = pd.get_dummies(train_labels)
#         self.__dummy_columns__ = self.__labels.columns
#         self.__labels = torch.tensor(self.__labels.values).to(torch.float32)
#         # 打乱顺序
#         print('data slicing...')
#         if shuffle:
#             # 按照获取比例抛弃部分数据集，打乱顺序后切换设备
#             data_len = len(self.__features)
#             indices = torch.randint(0, data_len, (data_len,))[:int(small_data * data_len)]
#             # self.__features__ = self.__features__[indices]
#             # self.__labels__ = self.__labels__[indices]
#             self.__features = torch.index_select(self.__features, dim=0, index=indices)
#             self.__labels = torch.index_select(self.__labels, dim=0, index=indices)
#         else:
#             self.__features = torch.index_select(
#                 self.__features, dim=0, index=torch.arange(len(self.__features) * small_data)
#             )
#             self.__labels = torch.index_select(
#                 self.__labels, dim=0, index=torch.arange(len(self.__labels) * small_data)
#             )
#         # 设备转移
#         self.__features = self.__features.to(self.device)
#         self.__labels = self.__labels.to(self.device)
#
#     def __check_path(self, path: str):
#         path_iter = os.walk(path)
#         _, folders, files = next(path_iter)
#         if 'images' in folders:
#             self.__images_dir__ = os.path.join(path, 'images')
#         else:
#             raise FileNotFoundError('该目录下无\"images\"文件夹！')
#         if 'train.csv' in files:
#             self.__train_dir__ = os.path.join(path, 'train.csv')
#         else:
#             raise FileNotFoundError('该目录下无\"train.csv\"文件！')
#
#     @property
#     def dummy(self) -> pd.Index:
#         return self.__dummy_columns__
#
#     @property
#     def train_data(self):
#         return self.__train_data__
#
#     @property
#     def valid_data(self):
#         return self.__valid_data__
#
#     @property
#     def device(self):
#         return self.__device__
#
#     def collate_fn(self, data: list):
#         # X, y = [], []
#         # for path, label in data:
#         #     X.append(read_img(path, required_shape=self.__required_shape__, mode='RGB'))
#         #     y.append(label)
#         # data = np.array(data)
#         # X, y = data[:, 0], data[:, 1]
#         X, y = np.array([x for x, _ in data]), torch.vstack([y for _, y in data])
#         # X = torch.from_numpy(X).to(torch.float32)
#         # y = torch.from_numpy(y).to(torch.float32)
#         # X = torch.as_tensor(X, dtype=torch.float32)
#         # y = torch.as_tensor(y, dtype=torch.float32)
#         # X, y = X.to(self.device), y.to(self.device)
#         X = torch.from_numpy(X).to(torch.float32).to(self.device).requires_grad_(True) \
#             if type(X) != torch.Tensor else X.to(torch.float32).to(self.device).requires_grad_(True)
#         y = torch.tensor(y, dtype=torch.float32, device=self.device, requires_grad=True) \
#             if type(y) != torch.Tensor else y.to(torch.float32).to(self.device).requires_grad_(True)
#         return X, y
#
#     def read_fn(self, index):
#         X = []
#         for path in index:
#             X.append(read_img(path, required_shape=self.__required_shape__, mode='RGB'))
#         # X = torch.from_numpy(np.array(X))
#         return np.array(X)

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
        if lazy:
            warnings.warn('将要进行懒加载。虽然这会降低内存消耗，但会极大地影响训练性能！')
            self.__read_data_lazily(small_data)
        else:
            self.__read_data(required_shape, small_data)
        assert len(self.__features) == len(self.__labels), '特征集和标签集长度须一致'
        # # 打乱数据后，切分训练集、测试集、验证集
        # self.__train_data__, _, self.__valid_data__ = split_data(
        #     self.__features, self.__labels, train, valid, shuffle=False
        # )

    def __read_data_lazily(self, small_data):
        print('reading indexes...')
        train_data = pd.read_csv(self.__train_dir__).values
        if small_data < 1:
            train_data = data_slicer(train_data, small_data)
        # 提取测试文件中的信息
        img_paths, labels = [], []
        for path, label in train_data:
            img_paths.append(os.path.join(self.__where, path))
            labels.append(label)
        # features_paths, labels = img_paths, train_data['label']
        # 将标签转化为独热编码
        print('getting dummies...')
        # labels = pd.get_dummies(labels.values)
        labels = pd.get_dummies(labels)
        self.__dummy_columns__ = labels.columns
        self.__labels = labels.values
        self.__features = np.array(img_paths)

    def __read_data(self, required_shape, small_data):
        train_features, train_labels = [], []
        train_data = pd.read_csv(self.__train_dir__).values
        if small_data < 1:
            train_data = data_slicer(train_data, small_data)
        # 取出文件中的数据
        # with tqdm(train_data.values, desc='reading data...', unit='img') as pbar:
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
        if self.__lazy:
            return LazyDataSet(self.__features, self.__labels, load_multiple, read_fn=self.read_fn)
        return DataSet(self.__features, self.__labels)

    @property
    def dummy(self) -> pd.Index:
        return self.__dummy_columns__

    # @property
    # def train_data(self):
    #     return self.__train_data__
    #
    # @property
    # def valid_data(self):
    #     return self.__valid_data__

    @property
    def device(self):
        return self.__device

    def collate_fn(self, data: list):
        pass
        # X, y = np.array([x for x, _ in data]), torch.vstack([y for _, y in data])
        # X = torch.from_numpy(X).to(torch.float32).to(self.device).requires_grad_(True) \
        #     if type(X) != torch.Tensor else X.to(torch.float32).to(self.device).requires_grad_(True)
        # y = torch.tensor(y, dtype=torch.float32, device=self.device, requires_grad=True) \
        #     if type(y) != torch.Tensor else y.to(torch.float32).to(self.device).requires_grad_(True)
        # return X, y

    def read_fn(self, index):
        X = []
        for path in index:
            X.append(read_img(path, required_shape=self.__required_shape, mode='RGB'))
        return np.array(X)


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
        # if lazy:
        #     warnings.warn('测试集正使用懒加载，虽然这会减少内存的消耗，但会极大地影响性能！')
        #     self.__read_data()
        self.__read_data()
        # self.__test_data__ = self.__test_data__.to(device)

    # def __read_data__(self, required_shape):
    #     test_data = pd.read_csv(self.__test_dir)
    #     features_paths = test_data['image']
    #     # # 取出文件中的数据
    #     # with tqdm(test_data.values, desc='reading test images...', unit='img') as pbar:
    #     #     for img_path in pbar:
    #     #         feature = read_img(os.path.join(self.__where__, img_path[0]), mode='RGB', required_shape=required_shape)
    #     #         test_features.append(feature)
    #     # self.__test_data__ = torch.from_numpy(np.array(test_features))
    #     # self.__test_data__ = self.__test_data__.to(torch.float32)
    #     # self.__raw_features__ = test_data
    #     self.__features__ = np.array(features_paths.values)

    def __read_data(self):
        print('reading indexes...')
        # test_data = pd.read_csv(self.__test_dir)
        # # 提取测试文件中的信息
        # img_paths = []
        # for path in test_data:
        #     img_paths.append(os.path.join(self.__where, path))
        # self.__features = np.array(img_paths)
        self.__features = pd.read_csv(self.__test_dir)

    # def __read_data(self, required_shape):
    #     features = []
    #     test_data = pd.read_csv(self.__test_dir).values
    #     # 取出文件中的数据
    #     with tqdm(test_data, desc='reading imgs...', unit='img') as pbar:
    #         for img_path in pbar:
    #             feature = read_img(os.path.join(self.__where, img_path), mode='RGB',
    #                                required_shape=required_shape)
    #             features.append(feature)
    #     print('turing data into tensor...')
    #     self.__features = torch.from_numpy(np.array(features))
    #     # 类型转移
    #     self.__features = self.__features.to(torch.float32)
    #     # 设备转移
    #     self.__features = self.__features.to(self.device)

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

    def __getitem__(self, item):
        return self.__features.iloc[item]

    @property
    def img_paths(self):
        return self.__features

    @property
    def imgs(self):
        return self.collate_fn(self.__features.values)
    # @property
    # def raw_features(self) -> pd.DataFrame:
    #     return self.__raw_features__

    # @property
    # def test_data(self) -> torch.Tensor:
    #     return self.__test_data__

    # @property
    # def features_shape(self):
    #     return (3, 224, 224)
