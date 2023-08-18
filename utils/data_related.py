import numpy as np
import torch
from PIL import Image
from typing import Tuple

from utils import tools


def single_argmax_accuracy(Y_HAT: torch.Tensor, Y: torch.Tensor) -> float:
    y_hat = torch.argmax(Y_HAT, dim=1)
    y = torch.argmax(Y, dim=1)
    cmp = (y_hat == y).type(Y.dtype)
    return float(sum(cmp))


def split_data(features: torch.Tensor, labels: torch.Tensor, train, test, valid=.0,
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
    test_len = int(data_len * test)
    # # 将高维特征数据打上id
    # features_ids = np.array([
    #     np.ones((1, *self.__features__.shape[2:])) * i
    #     for i in range(data_len)
    # ])
    # self.__features__ = np.concatenate((features_ids, self.__features__), 1)
    # 数据集打乱
    if shuffle:
        index = torch.randint(0, data_len, (data_len, ))
        features = features[index]
        labels = labels[index]
    # 数据集分割
    train_fea, valid_fea, test_fea = features.split((train_len, valid_len, test_len))
    train_labels, valid_labels, test_labels = labels.split((train_len, valid_len, test_len))
    return (train_fea, train_labels), (valid_fea, valid_labels), \
        (test_fea, test_labels)


def read_img(path: str, required_shape: Tuple[int, int] = None, mode: str = 'L') -> torch.Tensor:
    img_modes = ['L', 'RGB']
    assert mode in img_modes, f'不支持的图像模式{mode}！'
    img = Image.open(path).convert(mode)
    # img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # 若有要求shape，则进行resize，边缘填充黑条
    if required_shape:
        img = tools.resize_img(img, required_shape)
    img = torch.tensor(np.array(img))
    # 复原出通道。1表示样本数量维
    if mode == 'L':
        img_channels = 1
    elif mode == 'RGB':
        img_channels = 3
    img = img.reshape((1, img_channels, *img.shape[:2]))
    # print(path.split('/')[-1])
    # img = torch.hstack((path.split('/')[-1], img))
    return img

