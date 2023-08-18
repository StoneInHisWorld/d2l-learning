import os

import torch
from typing import Tuple
import pandas as pd
from tqdm import trange, tqdm

from utils import tools
from utils.data_related import read_img, split_data


class Leaves_Train:

    def __init__(self, where: str, train=0.8, valid=0.2, small_data=1., shuffle=True,
                 required_shape: Tuple[int, int] = None, device: torch.device = 'cpu'):
        self.__check_path__(where)
        self.__where__ = where
        self.__read_data__(required_shape)
        # 按照获取比例抛弃部分数据集，打乱顺序后切换设备
        data_len = len(self.__features__)
        indices = torch.randint(0, data_len, (data_len,))[:int(small_data * data_len)]
        self.__features__ = self.__features__[indices]
        self.__labels__ = self.__labels__[indices]
        self.__features__ = self.__features__.to(device)
        self.__labels__ = self.__labels__.to(device)
        assert len(self.__features__) == len(self.__labels__), '特征集和标签集长度须一致'
        self.__train_data__, _, self.__valid_data__ = split_data(
            self.__features__, self.__labels__, train, valid, shuffle=shuffle
        )

    # def __read_features__(self, required_shape=None) -> None:
    #     data = []
    #     path_iter = os.walk(self.__feature_dir__)
    #     for _, __, file_names in path_iter:
    #         file_names = sorted(
    #             file_names, key=lambda name: int(name.split(".")[0])
    #         )  # 给文件名排序！
    #         for fn in file_names:
    #             path = os.path.join(self.__feature_dir__, fn)
    #             data.append(
    #                 read_img(path, 1, required_shape=required_shape)
    #             )
    #     self.__features__ = torch.vstack(data).to(torch.float32)
    #
    # label_names = ['OAM1', 'OAM2']
    #
    # def __read_labels__(self, dummy=True) -> None:
    #     self.__labels__ = pd.read_csv(self.__label_dir__, names=Vortex.label_names)
    #     if dummy:
    #         self.__labels__ = pd.get_dummies(self.__labels__, columns=Vortex.label_names)
    #         self.__dummy_columns__ = self.__labels__.columns
    #     self.__labels__ = torch.tensor(self.__labels__.values, dtype=torch.float32)

    def __read_data__(self, required_shape):
        train_features, train_labels = [], []
        train_data = pd.read_csv(self.__train_dir__)
        # 取出文件中的数据
        with tqdm(train_data.values, desc='reading data...', unit='img') as pbar:
            for img_path, label in pbar:
            # for img_path, label in train_data.values:
                feature = read_img(os.path.join(self.__where__, img_path), mode='RGB', required_shape=required_shape)
                train_features.append(feature)
                train_labels.append(label)
        self.__features__ = torch.vstack(train_features).to(torch.float32)
        # 获取独热编码并转化为张量
        self.__labels__ = pd.get_dummies(pd.DataFrame(train_labels))
        self.__dummy_columns__ = self.__labels__.columns
        self.__labels__ = torch.tensor(self.__labels__.values).to(torch.float32)

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
        # if 'test.csv' in files:
        #     self.__test_dir__ = os.path.join(path, 'test.csv')
        # else:
        #     raise FileNotFoundError('该目录下无\"test.csv\"文件！')

    # def argmax_accuracy(self, Y_HAT, Y):
    #     assert Y_HAT.shape == Y.shape, f'预测值的维度{Y_HAT.shape}应与标签集维度{Y.shape}相等'
    #     # 将预测数据按照dummy列分组
    #     index_group = []
    #     for label_name in Vortex.label_names:
    #         label_dummy_index = [
    #             i for i, d in enumerate(self.__dummy_columns__)
    #             if label_name in d
    #         ]
    #         index_group.append(label_dummy_index)
    #     # 按组进行argmax，并比较
    #     cmps = []
    #     for group in index_group:
    #         y_hat_argmax = torch.argmax(Y_HAT[:, group], 1)
    #         y_argmax = torch.argmax(Y[:, group], 1)
    #         cmp = y_argmax == y_hat_argmax
    #         cmps = torch.vstack((cmps, cmp)) if len(cmps) > 0 else cmp
    #     # 计数全1行，即预测正确行
    #     correct = 0
    #     for cmp in cmps.T:
    #         correct += 1 if sum(cmp) == len(cmp) else 0
    #     return correct

    @property
    def dummy(self):
        return self.__dummy_columns__

    @property
    def train_data(self):
        return self.__train_data__

    @property
    def valid_data(self):
        return self.__valid_data__

    # @property
    # def test_data(self):
    #     return self.__test_data__


class Leaves_Test:

    def __init__(self, where: str, required_shape: Tuple[int, int] = None, device: torch.device = 'cpu'):
        self.__check_path__(where)
        self.__where__ = where
        self.__read_data__(required_shape)
        self.__test_data__ = self.__test_data__.to(device)

    # def __read_features__(self, required_shape=None) -> None:
    #     data = []
    #     path_iter = os.walk(self.__feature_dir__)
    #     for _, __, file_names in path_iter:
    #         file_names = sorted(
    #             file_names, key=lambda name: int(name.split(".")[0])
    #         )  # 给文件名排序！
    #         for fn in file_names:
    #             path = os.path.join(self.__feature_dir__, fn)
    #             data.append(
    #                 read_img(path, 1, required_shape=required_shape)
    #             )
    #     self.__features__ = torch.vstack(data).to(torch.float32)
    #
    # label_names = ['OAM1', 'OAM2']
    #
    # def __read_labels__(self, dummy=True) -> None:
    #     self.__labels__ = pd.read_csv(self.__label_dir__, names=Vortex.label_names)
    #     if dummy:
    #         self.__labels__ = pd.get_dummies(self.__labels__, columns=Vortex.label_names)
    #         self.__dummy_columns__ = self.__labels__.columns
    #     self.__labels__ = torch.tensor(self.__labels__.values, dtype=torch.float32)

    def __read_data__(self, required_shape):
        test_features = []
        test_data = pd.read_csv(self.__test_dir__)
        # 取出文件中的数据
        for img_path in test_data.iterrows():
            feature = read_img(os.path.join(self.__where__, img_path), mode='RGB', required_shape=required_shape)
            test_features.append(feature)
        self.__test_data__ = torch.vstack(test_features).to(torch.float32)

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

    # def argmax_accuracy(self, Y_HAT, Y):
    #     assert Y_HAT.shape == Y.shape, f'预测值的维度{Y_HAT.shape}应与标签集维度{Y.shape}相等'
    #     # 将预测数据按照dummy列分组
    #     index_group = []
    #     for label_name in Vortex.label_names:
    #         label_dummy_index = [
    #             i for i, d in enumerate(self.__dummy_columns__)
    #             if label_name in d
    #         ]
    #         index_group.append(label_dummy_index)
    #     # 按组进行argmax，并比较
    #     cmps = []
    #     for group in index_group:
    #         y_hat_argmax = torch.argmax(Y_HAT[:, group], 1)
    #         y_argmax = torch.argmax(Y[:, group], 1)
    #         cmp = y_argmax == y_hat_argmax
    #         cmps = torch.vstack((cmps, cmp)) if len(cmps) > 0 else cmp
    #     # 计数全1行，即预测正确行
    #     correct = 0
    #     for cmp in cmps.T:
    #         correct += 1 if sum(cmp) == len(cmp) else 0
    #     return correct

    @property
    def test_data(self):
        return self.__test_data__
