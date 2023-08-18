import pandas as pd
import torch


def kaggle_predict(net, raw_fea: pd.DataFrame, ripe_fea: torch.Tensor, dummy: pd.Index,
                   fea_colName: str = 'features', label_colName: str = 'label') -> None:
    # 未进行分类
    y_hat = torch.argmax(net(ripe_fea), 1)
    raw_fea[label_colName] = pd.Series(dummy[y_hat])
    # submission = pd.concat([raw_fea[fea_colName], raw_fea[label_colName]], axis=1)
    raw_fea.to_csv('submission.csv', index=False)