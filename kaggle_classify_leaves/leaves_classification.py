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
random_seed = 42
base_s = [2]
epochs_es = [10]
batch_sizes = [2, 4, 8, 16, 32, 64, 128, 256]
loss_es = ['entro']
lr_s = [1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
optim_str_s = ['sgd']
w_decay_s = [0.]

torch.random.manual_seed(random_seed)

print('collecting data...')
# 转移设备
device = tools.try_gpu(0)
# device = 'cpu'
# train_data = LeavesTrain('./classify-leaves', device=device)
# test_data = LeavesTest('./classify-leaves', device=device)
train_data = LeavesTrain('./classify-leaves', device=device, lazy=False)
test_data = LeavesTest('./classify-leaves', device=device)
# collate_fn = train_data.collate_fn
# read_fn = train_data.read_fn
acc_func = dr.single_argmax_accuracy

print('preprocessing...')
# 训练集与测试集封装以及数据转换，并对标签进行独热编码
dummies_column = train_data.dummy
train_ds = train_data.to_dataset()
train_sampler, valid_sampler = dr.split_data_(train_ds)
# train_ds = DataSet(
#     train_data.train_data[0], train_data.train_data[1]
# )
# valid_ds = DataSet(
#     train_data.valid_data[0], train_data.valid_data[1]
# )
del train_data

for base, epochs, batch_size, loss, lr, optim_str, w_decay in permutation(
        [], base_s, epochs_es, batch_sizes, loss_es, lr_s, optim_str_s, w_decay_s
):
    start = time.time()
    # train_iter = train_ds.to_loader(batch_size, shuffle=False, collate_fn=collate_func, lazy=True)
    # valid_iter = valid_ds.to_loader(shuffle=False, collate_fn=collate_func, lazy=True)
    train_iter = dr.to_loader(train_ds, batch_size, train_sampler)
    valid_iter = dr.to_loader(train_ds, sampler=valid_sampler)
    dataset_name = LeavesTrain.__name__

    print('constructing network...')
    in_channels = LeavesTrain.img_channels
    out_features = train_ds.label_shape[1]
    # TODO: 选择一个网络类型
    # net = VGG(in_channels, out_features, conv_arch=vgg.VGG_11, device=device)
    net = LeNet(in_channels, out_features, device=device)

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
        savefig_as=f'base{base} batch_size{batch_size} lr{lr} random_seed{random_seed} '
                   f'epochs{epochs}.jpg'
    )
    time_span = time.strftime('%H:%M:%S', time.gmtime(time.time() - start))
    tools.write_log(
        './log/alexnet_log.csv',
        net=net.__class__, epochs=epochs, batch_size=batch_size, loss=loss,
        lr=lr, random_seed=random_seed,
        train_l=sum(history['train_l']) / len(history['train_l']),
        train_acc=sum(history['train_acc']) / len(history['train_acc']),
        dataset=dataset_name, duration=time_span
    )
    print(f'train_acc = {history["train_acc"][-1] * 100:.2f}%, '
          f'train_l = {history["train_l"][-1]:.5f}\n')
    print(f'valid_acc = {history["valid_acc"][-1] * 100:.2f}%, '
          f'valid_l = {history["valid_l"][-1]:.5f}\n')

    print('predicting...')
    # kutils.kaggle_predict(net, test_data.raw_features, test_data.test_data, dummies_column,
    #                       fea_colName='image')
