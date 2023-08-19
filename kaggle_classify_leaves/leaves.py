import os
import warnings
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from utils.data_related import read_img, split_data
from utils.datasets import DataSet


class LeavesTrain:
    img_channels = 3

    def __init__(self, where: str, train=0.8, valid=0.2, small_data=1., shuffle=True,
                 lazy=True,
                 required_shape: Tuple[int, int] = None, device: torch.device = 'cpu'):
        """
        使用懒加载模式，只读取数据索引并存储，数据本身的加载留给DataLoader
        :param where: 全部数据存放的目录，即train.csv等文件的所属根目录
        :param train: 训练集比例
        :param valid: 验证集比例
        :param small_data: 选取数据的比例
        :param shuffle: 是否打乱数据集
        :param required_shape: 是否需要重塑读取图片的形状，None则为不需要，否则将图片重塑为指定元组形状
        :param device: 将数据迁移到所属设备上
        :param lazy: 是否进行懒加载
        """
        self.__where__ = where
        self.__required_shape__ = required_shape
        self.__device__ = device
        self.__lazy__ = lazy

        self.__check_path__(where)
        if lazy:
            warnings.warn('将要进行懒加载。虽然这会降低内存消耗，但会极大地影响训练性能！')
            self.__read_data_lazily__(small_data, shuffle)
        else:
            self.__read_data__(required_shape, small_data, shuffle)
        # # 按照获取比例抛弃部分数据集，打乱顺序后切换设备
        # data_len = len(self.__features__)
        # indices = torch.randint(0, data_len, (data_len, ))[:int(small_data * data_len)]
        # self.__features__ = self.__features__[indices]
        # self.__labels__ = self.__labels__[indices]
        # # self.__features__ = self.__features__.to(device)
        # # self.__labels__ = self.__labels__.to(device)
        assert len(self.__features__) == len(self.__labels__), '特征集和标签集长度须一致'
        # 打乱数据后，切分训练集、测试集、验证集
        self.__train_data__, _, self.__valid_data__ = split_data(
            self.__features__, self.__labels__, train, valid, shuffle=False
        )

    def __read_data_lazily__(self, small_data, shuffle=True):
        train_data = pd.read_csv(self.__train_dir__)
        # 提取测试文件中的信息
        img_paths = []
        for path, _ in train_data.values:
            img_paths.append(os.path.join(self.__where__, path))
        features_paths, labels = img_paths, train_data['label']
        # 将标签转化为独热编码
        print('getting dummies...')
        labels = pd.get_dummies(labels.values)
        self.__dummy_columns__ = labels.columns
        self.__labels__ = labels.values
        self.__features__ = np.array(img_paths)
        # 取出部分数据集并打乱
        if shuffle:
            # 按照获取比例抛弃部分数据集，打乱顺序后切换设备
            data_len = len(self.__features__)
            indices = torch.randint(0, data_len, (data_len,))[:int(small_data * data_len)]
            self.__features__ = self.__features__[indices]
            self.__labels__ = self.__labels__[indices]

    def __read_data__(self, required_shape, small_data, shuffle=True):
        train_features, train_labels = [], []
        train_data = pd.read_csv(self.__train_dir__)
        # 取出文件中的数据
        with tqdm(train_data.values, desc='reading data...', unit='img') as pbar:
            for img_path, label in pbar:
                feature = read_img(os.path.join(self.__where__, img_path), mode='RGB',
                                   required_shape=required_shape)
                train_features.append(feature)
                train_labels.append(label)
        self.__features__ = torch.from_numpy(np.array(train_features)).to(torch.float32)
        # 获取独热编码并转化为张量
        self.__labels__ = pd.get_dummies(train_labels)
        self.__dummy_columns__ = self.__labels__.columns
        self.__labels__ = torch.tensor(self.__labels__.values).to(torch.float32)
        # 打乱顺序
        if shuffle:
            # 按照获取比例抛弃部分数据集，打乱顺序后切换设备
            data_len = len(self.__features__)
            indices = torch.randint(0, data_len, (data_len,))[:int(small_data * data_len)]
            self.__features__ = self.__features__[indices]
            self.__labels__ = self.__labels__[indices]
        # 设备转移
        self.__features__ = self.__features__.to(self.device)
        self.__labels__ = self.__labels__.to(self.device)

    def __check_path__(self, path: str):
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

    @property
    def dummy(self) -> pd.Index:
        return self.__dummy_columns__

    @property
    def train_data(self):
        return self.__train_data__

    @property
    def valid_data(self):
        return self.__valid_data__

    @property
    def device(self):
        return self.__device__

    def collate_fn(self, data):
        X, y = [], []
        for path, label in data:
            X.append(read_img(path, required_shape=self.__required_shape__, mode='RGB'))
            y.append(label)
        X = torch.from_numpy(np.vstack(X)).to(torch.float32)
        y = torch.from_numpy(np.vstack(y)).to(torch.float32)
        X, y = X.to(self.device), y.to(self.device)
        return X, y


class LeavesTest:

    def __init__(self, where: str, required_shape: Tuple[int, int] = None, device: torch.device = 'cpu'):
        self.__check_path__(where)
        self.__where__ = where
        print('reading test_data...')
        self.__read_data__(required_shape)
        # self.__test_data__ = self.__test_data__.to(device)

    def __read_data__(self, required_shape):
        test_features = []
        test_data = pd.read_csv(self.__test_dir__)
        features_paths = test_data['image']
        # # 取出文件中的数据
        # with tqdm(test_data.values, desc='reading test images...', unit='img') as pbar:
        #     for img_path in pbar:
        #         feature = read_img(os.path.join(self.__where__, img_path[0]), mode='RGB', required_shape=required_shape)
        #         test_features.append(feature)
        # self.__test_data__ = torch.from_numpy(np.array(test_features))
        # self.__test_data__ = self.__test_data__.to(torch.float32)
        # self.__raw_features__ = test_data
        self.__features__ = np.array(features_paths.values)

    def __check_path__(self, path: str):
        path_iter = os.walk(path)
        _, folders, files = next(path_iter)
        if 'images' in folders:
            self.__images_dir__ = os.path.join(path, 'images')
        else:
            raise FileNotFoundError('该目录下无\"images\"文件夹！')
        if 'test.csv' in files:
            self.__test_dir__ = os.path.join(path, 'test.csv')
        else:
            raise FileNotFoundError('该目录下无\"test.csv\"文件！')

    # @property
    # def raw_features(self) -> pd.DataFrame:
    #     return self.__raw_features__

    @property
    def test_data(self) -> torch.Tensor:
        return self.__test_data__

    @property
    def features_shape(self):
        return (3, 224, 224)
