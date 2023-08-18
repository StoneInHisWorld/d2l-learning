import os
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from utils.data_related import read_img, split_data


class LeavesTrain:

    def __init__(self, where: str, train=0.8, valid=0.2, small_data=1., shuffle=True,
                 required_shape: Tuple[int, int] = None, device: torch.device = 'cpu'):
        self.__check_path__(where)
        self.__where__ = where
        self.__read_data__(required_shape)
        # 按照获取比例抛弃部分数据集，打乱顺序后切换设备
        data_len = len(self.__features__)
        indices = torch.randint(0, data_len, (data_len,))[:int(small_data * data_len)]
        # TODO: 寻找一种更加省内存的切片方式
        self.__features__ = self.__features__[indices]
        self.__labels__ = self.__labels__[indices]
        self.__features__ = self.__features__.to(device)
        self.__labels__ = self.__labels__.to(device)
        assert len(self.__features__) == len(self.__labels__), '特征集和标签集长度须一致'
        self.__train_data__, _, self.__valid_data__ = split_data(
            self.__features__, self.__labels__, train, valid, shuffle=shuffle
        )

    def __read_data__(self, required_shape):
        train_features, train_labels = [], []
        train_data = pd.read_csv(self.__train_dir__)
        # 取出文件中的数据
        with tqdm(train_data.values, desc='reading images...', unit='img') as pbar:
            for img_path, label in pbar:
                feature = read_img(
                    os.path.join(self.__where__, img_path), mode='RGB', required_shape=required_shape
                )
                # train_features = torch.vstack((train_features, feature)) if len(train_features) > 0 else feature
                train_features.append(feature)
                train_labels.append(label)
        self.__features__ = torch.from_numpy(np.array(train_features))
        # self.__features__ = torch.vstack(train_features)
        # self.__features__ = torch.from_numpy(np.array(train_features))
        # 获取独热编码并转化为张量
        del train_features, train_data
        print('getting dummies...')
        self.__labels__ = pd.get_dummies(train_labels)
        del train_labels
        self.__dummy_columns__ = self.__labels__.columns
        self.__labels__ = torch.from_numpy(self.__labels__.values)
        self.__features__ = self.__features__.type(torch.float32)
        self.__labels__ = self.__labels__.type(torch.float32)

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


class LeavesTest:

    def __init__(self, where: str, required_shape: Tuple[int, int] = None, device: torch.device = 'cpu'):
        self.__check_path__(where)
        self.__where__ = where
        print('reading test_data...')
        self.__read_data__(required_shape)
        self.__test_data__ = self.__test_data__.to(device)

    def __read_data__(self, required_shape):
        test_features = []
        test_data = pd.read_csv(self.__test_dir__)
        # 取出文件中的数据
        with tqdm(test_data.values, desc='reading test images...', unit='img') as pbar:
            for img_path in pbar:
                feature = read_img(os.path.join(self.__where__, img_path[0]), mode='RGB', required_shape=required_shape)
                test_features.append(feature)
        self.__test_data__ = torch.from_numpy(np.array(test_features))
        self.__test_data__ = self.__test_data__.to(torch.float32)
        self.__raw_features__ = test_data

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

    @property
    def raw_features(self) -> pd.DataFrame:
        return self.__raw_features__

    @property
    def test_data(self) -> torch.Tensor:
        return self.__test_data__
