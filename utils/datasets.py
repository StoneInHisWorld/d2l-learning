from typing import Iterable, Callable

import torch
from torch.utils.data import Dataset as torch_dataset, DataLoader


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

    def to_loader(self, batch_size: int = None, shuffle=True, collate_fn=None, **kwargs) -> DataLoader:
        """
        将数据集转化为DataLoader
        # :param load_multiple: 懒加载单次加载的倍数。懒加载每次读取数据量规定为`load_multiple * batch_size`。仅在`lazy = True`时有效
        # :param read_fn: 懒加载数据加载器读取方法。仅在`lazy = True`时有效
        # :param lazy: 启用懒加载DataLoader
        :param collate_fn: 数据预处理函数
        :param batch_size: DataLoader每次供给的数据量。默认为整个数据集
        :param shuffle: 是否打乱
        :param kwargs: Dataloader额外参数
        :return: DataLoader对象
        """
        if not batch_size:
            batch_size = self.feature_shape[0]
        # if lazy:
        #     return LazyDataLoader(
        #         self, read_fn, batch_size, load_multiple=load_multiple, shuffle=shuffle,
        #         collate_fn=collate_fn, **kwargs
        #     )
        # else:
        return DataLoader(
            self, batch_size, shuffle=shuffle, collate_fn=collate_fn, **kwargs
        )

    def get_subset(self, indices: Iterable):
        return DataSet(self[indices][0], self[indices][1])

    @property
    def feature_shape(self):
        return self.__features__.shape

    @property
    def label_shape(self):
        return self.__labels__.shape


class LazyDataLoader:
    def __init__(self, index_dataset: DataSet, read_fn, batch_size: int = None, load_multiple: int = 1,
                 shuffle=True, collate_fn=None,
                 **kwargs):
        self.__batch_size__ = batch_size
        self.__shuffle__ = shuffle
        self.__collate_fn__ = collate_fn
        self.__read_fn__ = read_fn
        self.__kwargs__ = kwargs

        self.__index_loader__ = index_dataset.to_loader(batch_size * load_multiple, shuffle, collate_fn)
        pass

    def __iter__(self):
        for index, label in self.__index_loader__:
            batch_loader = DataSet(self.__read_fn__(index), label).to_loader(
                self.__batch_size__, self.__shuffle__, self.__collate_fn__, **self.__kwargs__
            )
            for X, y in batch_loader:
                yield X, y

    # def load_data(self):
    #     """
    #     将单个loader所涉及的数据加载到内存中，打包成DataLoader
    #     :return:
    #     """
    #     X, y = [], []
    #     for path, label in self.__index_loader__:
    #         X.append(read_img(path, required_shape=self.__required_shape__, mode='RGB'))
    #         y.append(label)
    #     X = torch.from_numpy(np.vstack(X)).to(torch.float32)
    #     y = torch.from_numpy(np.vstack(y)).to(torch.float32)
    #     X, y = X.to(self.device), y.to(self.device)
