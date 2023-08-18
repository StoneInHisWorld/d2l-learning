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

    def to_loader(self, batch_size: int = None, shuffle=True, **kwargs) -> DataLoader:
        """
        将数据集转化为DataLoader
        :param batch_size: DataLoader每次供给的数据量。默认为整个数据集
        :param shuffle: 是否打乱
        :param kwargs: Dataloader额外参数
        :return: DataLoader对象
        """
        if not batch_size:
            batch_size = self.feature_shape[0]
        return DataLoader(self, batch_size, shuffle=shuffle, **kwargs)

    @property
    def feature_shape(self):
        return self.__features__.shape

    @property
    def label_shape(self):
        return self.__labels__.shape


class Vortex:

    def __init__(self, where: str, train=0.8, test=0.2, valid=0.,
                 small_data=1., shuffle=True,
                 required_shape: Tuple[int, int] = None, device: torch.device = 'cpu'):
        self.__check_path__(where)
        self.__read_features__(required_shape)
        self.__read_labels__()
        # 按照获取比例抛弃部分数据集，打乱顺序后切换设备
        data_len = len(self.__features__)
        indices = torch.randint(0, data_len, (data_len,))[:int(small_data * data_len)]
        self.__features__ = self.__features__[indices]
        self.__labels__ = self.__labels__[indices]
        self.__features__ = self.__features__.to(device)
        self.__labels__ = self.__labels__.to(device)
        assert len(self.__features__) == len(self.__labels__), '特征集和标签集长度须一致'
        self.__train_data__, self.__valid_data__, self.__test_data__ = \
            split_data(
                self.__features__, self.__labels__,
                train, test, valid, shuffle=shuffle)

    def __read_features__(self, required_shape=None) -> None:
        data = []
        path_iter = os.walk(self.__feature_dir__)
        for _, __, file_names in path_iter:
            file_names = sorted(
                file_names, key=lambda name: int(name.split(".")[0])
            )  # 给文件名排序！
            for fn in file_names:
                path = os.path.join(self.__feature_dir__, fn)
                data.append(
                    read_img(path, required_shape=required_shape)
                )
        self.__features__ = torch.vstack(data).to(torch.float32)

    label_names = ['OAM1', 'OAM2']

    def __read_labels__(self, dummy=True) -> None:
        self.__labels__ = pd.read_csv(self.__label_dir__, names=Vortex.label_names)
        if dummy:
            self.__labels__ = pd.get_dummies(self.__labels__, columns=Vortex.label_names)
            self.__dummy_columns__ = self.__labels__.columns
        self.__labels__ = torch.tensor(self.__labels__.values, dtype=torch.float32)

    def __check_path__(self, path: str):
        path_iter = os.walk(path)
        _, folders, __ = next(path_iter)
        if 'vortex' in folders:
            path = os.path.join(path, 'vortex')
            path_iter = os.walk(path)
            for _, folders, file_names in path_iter:
                if '0823SPECKLE' in folders and 'labels.csv' in file_names:
                    self.__feature_dir__ = os.path.join(path, '0823SPECKLE')
                    self.__label_dir__ = os.path.join(path, 'labels.csv')
                    break
                else:
                    raise FileNotFoundError('该目录下尚未找到\"0823SPECKLE\"或\"labels.csv\"文件夹！')
        else:
            raise FileNotFoundError('该目录下无\"vortex\"文件夹！')

    def argmax_accuracy(self, Y_HAT, Y):
        assert Y_HAT.shape == Y.shape, f'预测值的维度{Y_HAT.shape}应与标签集维度{Y.shape}相等'
        # 将预测数据按照dummy列分组
        index_group = []
        for label_name in Vortex.label_names:
            label_dummy_index = [
                i for i, d in enumerate(self.__dummy_columns__)
                if label_name in d
            ]
            index_group.append(label_dummy_index)
        # 按组进行argmax，并比较
        cmps = []
        for group in index_group:
            y_hat_argmax = torch.argmax(Y_HAT[:, group], 1)
            y_argmax = torch.argmax(Y[:, group], 1)
            cmp = y_argmax == y_hat_argmax
            cmps = torch.vstack((cmps, cmp)) if len(cmps) > 0 else cmp
        # 计数全1行，即预测正确行
        correct = 0
        for cmp in cmps.T:
            correct += 1 if sum(cmp) == len(cmp) else 0
        return correct

    @property
    def dummy(self):
        return self.__dummy_columns__

    @property
    def train_data(self):
        return self.__train_data__

    @property
    def valid_data(self):
        return self.__valid_data__

    @property
    def test_data(self):
        return self.__test_data__
