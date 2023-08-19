from torch.utils.data import DataLoader, RandomSampler

from utils.datasets import DataSet


class LazyDataLoader:
    def __init__(self, index_dataset: DataSet, batch_size: int = None, load_multiple: int = 1,
                 shuffle=True, collate_fn=None,
                 **kwargs):
        self.__dataset__ = index_dataset
        self.__batch_size__ = batch_size
        self.__load_multiple__ = load_multiple
        pass

    def __iter__(self):


    def loader_supply(self):
        """
        将单个loader所涉及的数据加载到内存中，打包成DataLoader
        :return:
        """
        # full_sampler = RandomSampler(self.__dataset__, replacement=False,
        #                              num_samples=self.__batch_size__ * self.__load_multiple__)
        indexes_slices = len(self.__dataset__)
        for indexes in full_sampler:
            # 打包一个局部DataLoader
            loader = DataLoader(self.__dataset__.get_subset(indexes))

