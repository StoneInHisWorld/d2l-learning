from utils.datasets import DataSet


class LazyDataLoader:
    def __init__(self, index_dataset: DataSet, read_fn, batch_size: int = None, load_multiple: int = 1,
                 shuffle=True, collate_fn=None,
                 **kwargs):
        self.__batch_size__ = batch_size
        self.__multiple__ = load_multiple
        self.__shuffle__ = shuffle
        self.__collate_fn__ = collate_fn
        self.__read_fn__ = read_fn
        self.__kwargs__ = kwargs

        self.__index_loader__ = index_dataset.to_loader(batch_size * load_multiple, shuffle)
        pass

    def __iter__(self):
        for index, label in self.__index_loader__:
            batch_loader = DataSet(self.__read_fn__(index), label).to_loader(
                self.__batch_size__, self.__shuffle__, self.__collate_fn__, **self.__kwargs__
            )
            for X, y in batch_loader:
                yield X, y

    def __len__(self):
        return len(self.__index_loader__) * self.__multiple__
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