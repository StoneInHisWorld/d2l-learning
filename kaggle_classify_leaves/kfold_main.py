import random
import time

import numpy as np
import torch

import networks.common_layers
import utils.data_related as dr
import utils.tools as tools
from leaves import LeavesTrain, LeavesTest
from networks.alexnet import AlexNet
from networks.lenet import LeNet
from utils.datasets import DataSet
from utils.tools import permutation
import utils.kaggle_utils as kutils

# 调参面板
exp_no = 2
random_seed = 42
data_portion = 1.0
Net = AlexNet
k_s = [10]
base_s = [2]
epochs_es = [5]
batch_sizes = [4, 8, 16, 32, 64, 128, 256]
loss_es = ['entro']
lr_s = [1e-3]
optim_str_s = ['adam']
w_decay_s = [0.]

torch.random.manual_seed(random_seed)

print('collecting data...')
# 转移设备
device = tools.try_gpu(0)
# device = 'cpu'
train_data = LeavesTrain(
    './classify-leaves', device=device, lazy=False, small_data=data_portion,
    required_shape=Net.required_shape
)
test_data = LeavesTest('./classify-leaves', device=device, required_shape=Net.required_shape)
acc_func = dr.single_argmax_accuracy


def my_reshape(data, part=4):
    d_len = len(data)
    p_len = d_len // part
    indices = np.split(np.arange(d_len), (p_len, p_len * 2, p_len * 3))
    for i in indices:
        data[i] = networks.common_layers.Reshape(Net.required_shape)(data[i])


features_preprocess = [
    # torch.nn.functional.batch_norm,
    # networks.common_layers.Reshape(Net.required_shape)
    # my_reshape
]
labels_process = []

print('preprocessing...')
# 训练集封装，并生成训练集和验证集的sampler
dummies_column = train_data.dummy
train_ds = train_data.to_dataset()
del train_data

for k, base, epochs, batch_size, loss, lr, optim_str, w_decay in permutation(
        [], k_s, base_s, epochs_es, batch_sizes, loss_es, lr_s, optim_str_s, w_decay_s
):
    start = time.time()
    train_ds.apply(features_preprocess, labels_process)
    sampler_iter = dr.k_fold_split(train_ds, k)
    train_loaders = (
        (dr.to_loader(train_ds, batch_size, sampler=train_sampler),
         dr.to_loader(train_ds, len(valid_sampler) // 3, sampler=valid_sampler))
        for train_sampler, valid_sampler in sampler_iter
    )
    dataset_name = LeavesTrain.__name__

    print('constructing network...')
    in_channels = LeavesTrain.img_channels
    out_features = train_ds.label_shape[1]
    # TODO: 选择一个网络类型
    # net = VGG(in_channels, out_features, conv_arch=vgg.VGG_11, device=device)
    # net = LeNet(in_channels, out_features, device=device)
    net = Net(in_channels, out_features, device=device)

    # 构建网络
    optimizer = tools.get_optimizer(net, optim_str, lr, w_decay)
    loss = tools.get_loss(loss)
    optimizer_name = optimizer.__class__.__name__

    print(f'training on {device}...')
    history = net.train_with_k_fold(
        train_loaders, optimizer=optimizer, num_epochs=epochs, ls_fn=loss, acc_fn=acc_func
    )
    del train_loaders, optimizer

    print('plotting...')
    tools.plot_history(
        history, xlabel='num_epochs', ylabel=f'loss({loss})', mute=True,
        title=f'dataset: {dataset_name} optimizer: {optimizer_name}\n'
              f'net: {net.__class__.__name__}',
        savefig_as=f'./imgs/base{base} batch_size{batch_size} lr{lr} random_seed{random_seed} '
                   f'epochs{epochs}.jpg'
    )
    time_span = time.strftime('%H:%M:%S', time.gmtime(time.time() - start))
    tools.write_log(
        './log/alexnet_log.csv',
        net=net.__class__, exp_no=exp_no, epochs=epochs, batch_size=batch_size,
        loss=loss, lr=lr, random_seed=random_seed,
        train_l=sum(history['train_l']) / len(history['train_l']),
        train_acc=sum(history['train_acc']) / len(history['train_acc']),
        valid_l=sum(history['valid_l']) / len(history['valid_l']),
        valid_acc=sum(history['valid_acc']) / len(history['valid_acc']),
        dataset=dataset_name, duration=time_span
    )
    print(f'train_acc = {history["train_acc"][-1] * 100:.2f}%, '
          f'train_l = {history["train_l"][-1]:.5f}')
    print(f'valid_acc = {history["valid_acc"][-1] * 100:.2f}%, '
          f'valid_l = {history["valid_l"][-1]:.5f}\n')
    del history

    print('predicting...')
    ripe_fea = test_data.imgs
    for call in features_preprocess:
        ripe_fea = call(ripe_fea)
    kutils.kaggle_predict(net, test_data.img_paths, ripe_fea, dummies_column, split=3)
    del net
