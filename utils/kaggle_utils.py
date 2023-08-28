import pandas as pd
import torch
from torch.utils.data import DataLoader

from networks.basic_nn import BasicNN


def kaggle_predict(net: BasicNN, raw_fea: pd.DataFrame, ripe_fea: torch.Tensor, dummy: pd.Index,
                   label_colName: str = 'label', split: int = 1) -> None:
    """
    使用网络进行预测，并形成提交结果集submission.csv，存储在当前目录
    :param net: 训练好的神经网络
    :param raw_fea: 原始文件test.csv中存储的特征读取路径
    :param ripe_fea: 特征数据实际内容
    :param dummy: 标签类别集
    :param label_colName: 要求输出的标签列名
    :return: None
    """
    feature_iter = iter(torch.split(ripe_fea, split))
    preds = net.predict_(feature_iter)
    y_hat = torch.argmax(preds, 1).cpu()
    raw_fea[label_colName] = pd.Series(dummy[y_hat])
    raw_fea.to_csv('submission.csv', index=False)
