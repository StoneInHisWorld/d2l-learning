import time

import torch

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
base_s = [2]
epochs_es = [300]
batch_sizes = [8, 16, 32, 64, 128]
loss_es = ['entro']
lr_s = [1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3]
optim_str_s = ['adam']
w_decay_s = [0.]

torch.random.manual_seed(random_seed)

print('collecting data...')
# 转移设备
device = tools.try_gpu(0)
# device = 'cpu'
train_data = LeavesTrain('./classify-leaves', device=device, lazy=False, small_data=1)
test_data = LeavesTest('./classify-leaves', device=device)
acc_func = dr.single_argmax_accuracy

print('preprocessing...')
# 训练集封装，并生成训练集和验证集的sampler
dummies_column = train_data.dummy
train_ds = train_data.to_dataset()
train_sampler, valid_sampler = dr.split_data(train_ds)
del train_data

for base, epochs, batch_size, loss, lr, optim_str, w_decay in permutation(
        [], base_s, epochs_es, batch_sizes, loss_es, lr_s, optim_str_s, w_decay_s
):
    start = time.time()
    train_iter = dr.to_loader(train_ds, batch_size, sampler=train_sampler)
    valid_iter = dr.to_loader(train_ds, sampler=valid_sampler)
    dataset_name = LeavesTrain.__name__

    print('constructing network...')
    in_channels = LeavesTrain.img_channels
    out_features = train_ds.label_shape[1]
    # TODO: 选择一个网络类型
    # net = VGG(in_channels, out_features, conv_arch=vgg.VGG_11, device=device)
    # net = LeNet(in_channels, out_features, device=device)
    net = AlexNet(in_channels, out_features, device=device)

    # 构建网络
    optimizer = tools.get_optimizer(net, optim_str, lr, w_decay)
    loss = tools.get_loss(loss)

    print(f'training on {device}...')
    history = net.train_(
        train_iter, optimizer=optimizer, num_epochs=epochs, ls_fn=loss,
        acc_fn=acc_func, valid_iter=valid_iter
    )
    print('plotting...')
    tools.plot_history(
        history, xlabel='num_epochs', ylabel=f'loss({loss})', mute=True,
        title=f'dataset: {dataset_name} optimizer: {optimizer.__class__.__name__}\n'
              f'net: {net.__class__.__name__}',
        savefig_as=f'./imgs/base{base} batch_size{batch_size} lr{lr} random_seed{random_seed} '
                   f'epochs{epochs}.jpg'
    )
    time_span = time.strftime('%H:%M:%S', time.gmtime(time.time() - start))
    tools.write_log(
        './log/alexnet_log.csv',
        net=net.__class__, epochs=epochs, batch_size=batch_size, loss=loss,
        lr=lr, random_seed=random_seed,
        train_l=sum(history['train_l']) / len(history['train_l']),
        train_acc=sum(history['train_acc']) / len(history['train_acc']),
        dataset=dataset_name, duration=time_span, exp_no=exp_no
    )
    print(f'train_acc = {history["train_acc"][-1] * 100:.2f}%, '
          f'train_l = {history["train_l"][-1]:.5f}')
    print(f'valid_acc = {history["valid_acc"][-1] * 100:.2f}%, '
          f'valid_l = {history["valid_l"][-1]:.5f}\n')

    print('predicting...')
    kutils.kaggle_predict(net, test_data.img_paths, test_data.imgs, dummies_column)
