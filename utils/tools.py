import PIL.Image
import pandas as pd
import torch
from PIL import Image as IMAGE
from PIL.Image import Image
from matplotlib import pyplot as plt
from torch import cuda, nn as nn
from torch.nn import init as init
from typing import Tuple

optimizers = ['sgd', 'adam']
loss_es = ['l1', 'entro', 'mse', 'huber']


def write_log(path: str, **kwargs):
    assert path.endswith('.csv'), f'日志文件格式为.csv，将要写入的文件名为{path}'
    try:
        file_data = pd.read_csv(path)
    except Exception as _:
        file_data = pd.DataFrame([])
    item_data = pd.DataFrame([kwargs])
    if len(file_data) == 0:
        file_data = pd.DataFrame(item_data)
    else:
        file_data = pd.concat([file_data, item_data], axis=0)
    file_data.to_csv(path, index=False)


def plot_history(history, mute=False, title=None, xlabel=None,
                 ylabel=None, savefig_as=None):
    for k, log in history:
        plt.plot(range(len(log)), log, label=k)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if title:
        plt.title(title)
    if savefig_as:
        plt.savefig(savefig_as)
    plt.legend()
    if not mute:
        plt.show()


def try_gpu(i=0):
    """
    获取一个GPU
    :param i: GPU编号
    :return: 第i号GPU。若GPU不可用，则返回CPU
    """
    if cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def permutation(res, *args):
    if len(args) == 0:
        yield res
    elif len(args) == 1:
        for arg in args[0]:
            yield res + [arg]
    else:
        for arg in args[0]:
            for p in permutation(res + [arg], *args[1:]):  # 不能直接yield本函数，否则不会执行
                yield p


def get_optimizer(net: torch.nn.Module, optim_str, lr=0.1, w_decay=0., momentum=0.):
    assert optim_str in optimizers, f'不支持优化器{optim_str}, 支持的优化器包括{optimizers}'
    if optim_str == 'sgd':
        return torch.optim.SGD(
            net.parameters(),
            lr=lr,
            weight_decay=w_decay,
            momentum=momentum
        )
    elif optim_str == 'adam':
        return torch.optim.Adam(
            net.parameters(),
            lr=lr,
            weight_decay=w_decay
        )


def get_loss(loss_str: str = 'mse'):
    assert loss_str in loss_es, \
        f'不支持激活函数{loss_str}, 支持的优化器包括{loss_es}'
    if loss_str == 'l1':
        return nn.L1Loss()
    elif loss_str == 'entro':
        return nn.CrossEntropyLoss()
    elif loss_str == 'mse':
        return nn.MSELoss()
    elif loss_str == 'huber':
        return nn.HuberLoss()


def init_wb(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        init.xavier_uniform_(m.weight)
        init.zeros_(m.bias)
        # init.uniform_(m.weight)
        # init.zeros_(m.bias)


def resize_img(image: Image, required_shape: Tuple[int, int], img_mode='L') -> Image:
    # ------------------------------#
    #   获得图像的高宽与目标高宽
    #   code from: Bubbliiiing
    # ------------------------------#
    iw, ih = image.size
    h, w = required_shape

    # 长边放缩比例
    scale = min(w / iw, h / ih)
    # 计算新图片shape
    new_w = int(iw * scale)
    new_h = int(ih * scale)
    # 计算图片缺失shape
    dx = (w - new_w) // 2
    dy = (h - new_h) // 2
    # ---------------------------------#
    #   将图像多余的部分加上黑条
    # ---------------------------------#
    image = image.resize((new_w, new_h), IMAGE.BICUBIC)
    new_image = IMAGE.new(img_mode, (w, h), 0)
    new_image.paste(image, (dx, dy))

    return new_image
