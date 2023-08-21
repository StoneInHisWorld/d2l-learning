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
epochs_es = [200]
batch_sizes = [256]
loss_es = ['entro']
lr_s = [1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
optim_str_s = ['sgd']
w_decay_s = [0.]

torch.random.manual_seed(random_seed)

print('collecting data...')
# 转移设备
# device = tools.try_gpu(0)
device = 'cpu'
train_data = LeavesTrain('./classify-leaves', device=device)
test_data = LeavesTest('./classify-leaves', device=device)
collate_func = train_data.collate_fn
acc_func = utils.data_related.single_argmax_accuracy

print('preprocessing...')
# 训练集与测试集封装以及数据转换，并对标签进行独热编码
dummies_column = train_data.dummy
train_ds = DataSet(
    train_data.train_data[0], train_data.train_data[1]
)
valid_ds = DataSet(
    train_data.valid_data[0], train_data.valid_data[1]
)
del train_data

for base, epochs, batch_size, loss, lr, optim_str, w_decay in permutation(
        [], base_s, epochs_es, batch_sizes, loss_es, lr_s, optim_str_s, w_decay_s
):
    # train_iter = train_ds.to_loader(batch_size, shuffle=False, collate_fn=collate_func, lazy=True)
    # valid_iter = valid_ds.to_loader(shuffle=False, collate_fn=collate_func, lazy=True)
    train_iter = dr.to_loader(train_ds, batch_size, lazy=True, read_fn=LeavesTrain.read_fn, load_multiple=5)
    valid_iter = dr.to_loader(train_ds, lazy=True, read_fn=LeavesTrain.read_fn, load_multiple=5)
    dataset_name = LeavesTrain.__name__

    print('constructing network...')
    in_channels = LeavesTrain.img_channels
    out_features = train_ds.label_shape[1]
    # TODO: 选择一个网络类型
    # net = VGG(in_channels, out_features, conv_arch=vgg.VGG_11, device=device)
    net = AlexNet(in_channels, out_features, device=device)

    # 构建网络
    optimizer = tools.get_optimizer(net, optim_str, lr, w_decay)
    loss = tools.get_loss(loss)

    print('training...')
    history = net.train_(
        train_iter, optimizer=optimizer, num_epochs=epochs, loss=loss,
        acc_func=acc_func
    )

    print('plotting...')
    tools.plot_history(
        history, xlabel='num_epochs', ylabel=f'loss({loss})', mute=False,
        title=f'dataset: {dataset_name} optimizer: {optimizer.__class__.__name__}\n'
              f'net: {net.__class__.__name__}',
        savefig_as=f'base{base} batch_size{batch_size} lr{lr} random_seed{random_seed} '
                   f'epochs{epochs}.jpg'
    )

    print('predicting...')
    kutils.kaggle_predict(net, test_data.raw_features, test_data.test_data, dummies_column,
                          fea_colName='image')
