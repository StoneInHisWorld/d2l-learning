import numpy as np
import torch

import utils.data_related as dr
import utils.tools as tools
from leaves import LeavesTrain, LeavesTest
from networks.nets.alexnet import AlexNet
from utils.hypa_control import ControlPanel

# 调参面板
Net = AlexNet
# 调参面板
cp = ControlPanel(LeavesTrain, 'leaves_hyper_params.json', 'settings.json',
                  'log/alexnet_log.csv')
torch.random.manual_seed(cp.running_randomseed)
device = cp.running_device
data_portion = cp.running_dataportion

print('collecting data...')
Net = AlexNet
# 转移设备
train_data = LeavesTrain(
    './classify-leaves', device=device, lazy=False, small_data=data_portion,
    required_shape=Net.required_shape
)
test_data = LeavesTest('./classify-leaves', device=device, required_shape=Net.required_shape)
acc_func = dr.single_argmax_accuracy
dataset_name = train_data.__class__.__name__

print('preprocessing...')
features_preprocess = [
    # LeavesTrain.features_preprocess
    # networks.common_layers.Reshape(Net.required_shape)
    # my_reshape
]
labels_process = []
# 训练集封装，并生成训练集和验证集的sampler
dummies_column = train_data.dummy
train_ds = train_data.to_dataset()
del train_data
for trainer in cp:
    with trainer as hps:
        k, base, epochs, batch_size, ls_fn, lr, optim_str, w_decay = hps
        device = cp.running_device
        exp_no = cp.running_expno

        # 数据准备工作
        train_ds.to(device)
        test_data.to(device)
        train_ds.apply(features_preprocess, labels_process)
        sampler_iter = dr.k_fold_split(train_ds, k)
        train_loaders = (
            (dr.to_loader(train_ds, batch_size, sampler=train_sampler),
             dr.to_loader(train_ds, len(valid_sampler) // 3, sampler=valid_sampler))
            for train_sampler, valid_sampler in sampler_iter
        )

        print(f'constructing a/an {Net.__name__}...')
        # 构建网络
        in_channels = LeavesTrain.img_channels
        out_features = train_ds.label_shape[1]
        # TODO: 选择一个网络类型
        net = Net(in_channels, out_features, device=device, with_checkpoint=False)

        print(f'training on {device}...')
        # 进行训练准备
        optimizer = tools.get_optimizer(net, optim_str, lr, w_decay)
        ls_fn = tools.get_loss(ls_fn)
        optimizer_name = optimizer.__class__.__name__
        history = net.train_with_k_fold(
            train_loaders, optimizer=optimizer, num_epochs=epochs, ls_fn=ls_fn, acc_fn=acc_func,
            k=k
        )
        del train_loaders

        print('plotting...')
        train_acc, train_l = history["train_acc"][-1], history["train_l"][-1]
        try:
            valid_acc, valid_l = history["valid_acc"][-1], history["valid_l"][-1]
        except Exception as _:
            valid_acc, valid_l = np.nan, np.nan
        print(f'train_acc = {train_acc * 100:.3f}%, train_l = {train_l:.5f}, '
              f'valid_acc = {valid_acc * 100:.3f}%, valid_l = {valid_l:.5f}')
        cp.plot_history(
            history, xlabel='num_epochs', ylabel=f'loss({ls_fn.__class__.__name__})',
            title=f'dataset: {dataset_name} optimizer: {optimizer.__class__.__name__}\n'
                  f'net: {net.__class__.__name__}',
            save_path=f'log/imgs/alexnet/{exp_no}.jpg'
        )
        trainer.add_logMsg(
            True,
            train_l=train_l, train_acc=train_acc, valid_l=valid_l, valid_acc=valid_acc,
            exp_no=exp_no, dataset=dataset_name,
            random_seed=cp.running_randomseed, data_portion=cp.running_dataportion
        )
        # print(f'train_acc = {history["train_acc"][-1] * 100:.2f}%, '
        #       f'train_l = {history["train_l"][-1]:.5f}')
        # print(f'valid_acc = {history["valid_acc"][-1] * 100:.2f}%, '
        #       f'valid_l = {history["valid_l"][-1]:.5f}\n')
        del ls_fn, optimizer, history

        # print('predicting...')
        # ripe_fea = test_data.imgs
        # for call in features_preprocess:
        #     ripe_fea = call(ripe_fea)
        # kutils.kaggle_predict(net, test_data.img_paths, ripe_fea, dummies_column, split=3)
        # del net
