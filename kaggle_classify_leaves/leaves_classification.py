import time

import torch
import torchvision
import torchvision.datasets as ds
import torch.nn.functional as F
from torchsummary import summary
from leaves import Leaves_Train, Leaves_Test

import utils.data_related
import utils.tools as tools
from networks import vgg
from networks.alexnet import AlexNet
from networks.lenet import LeNet
from networks.vgg import VGG
from utils.data_related import single_argmax_accuracy
from utils.datasets import DataSet
from utils.tools import permutation

# 调参面板
datasets = ['leaves']
random_seeds = 42
base_s = [2]
epochs_es = [3]
batch_sizes = [256]
loss_es = ['entro']
lr_s = [1e-2]
optim_str_s = ['sgd']
w_decay_s = [0.]

torch.random.manual_seed(random_seeds)

print('collecting data...')
# 读取FashionMNIST数据集
# 转移设备
device = tools.try_gpu(0)
train_data = Leaves_Train('./classify-leaves', device=device)
test_data = Leaves_Test('./classify-leaves', device=device).test_data
acc_func = utils.data_related.single_argmax_accuracy

print('preprocessing...')
features_shape = train_data.train_data.shape[-2:]
# 训练集与测试集封装以及数据转换，并对标签进行独热编码
train_ds = DataSet(
    train_data.train_data[0], train_data.train_data[1]
)
valid_ds = DataSet(
    train_data.valid_data[0], train_data.valid_data[1]
)

for dataset, base, epochs, batch_size, loss, lr, optim_str, w_decay in permutation(
    [], datasets, base_s, epochs_es, batch_sizes, loss_es, lr_s, optim_str_s, w_decay_s
):
    train_iter = train_ds.to_loader(batch_size)
    valid_iter = valid_ds.to_loader()

    print('constructing network...')
    # 构建网络
    in_channels = train_ds.feature_shape[1]
    out_features = train_ds.label_shape[1]
    # TODO: 选择一个网络类型
    # net = VGG(in_channels, out_features, conv_arch=vgg.VGG_11, device=device)
    net = LeNet(in_channels, out_features, device=device)
    try:
        summary(net, input_size=(1, 28, 28), batch_size=batch_size)
    except Exception as _:
        print(net)

    optimizer = tools.get_optimizer(net, optim_str, lr, w_decay)
    loss = tools.get_loss(loss)
    print('training...')
    history = net.train_(
        train_iter, optimizer=optimizer, num_epochs=epochs, loss=loss,
        acc_func=acc_func, valid_iter=valid_iter
    )
    print('testing...')
    preds = net(test_data)
    # print(f'test_acc = {test_acc * 100:.2f}%, test_ls = {test_ls}')
    # tools.plot_history(
    #     history, xlabel='num_epochs',
    #     ylabel=f'train_loss({loss})',
    #     mute=False
    # )
