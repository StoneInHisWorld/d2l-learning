from typing import Iterable, Sized, Any, Callable

import torch
import torch.nn as nn
from torch.nn import Module
from tqdm import tqdm, trange

from utils.accumulator import Accumulator
from utils.data_related import single_argmax_accuracy
from utils.datasets import DataSet
from utils.history import History
from utils.tools import init_wb


class BasicNN(nn.Sequential):
    required_shape = (-1,)

    def __init__(self, device, *args: Module) -> None:
        super().__init__(*args)
        self.apply(init_wb)
        self.apply(lambda m: m.to(device))
        self.__device__ = device

    def __str__(self):
        return '网络结构：\n' + super().__str__() + '\n所处设备：' + str(self.__device__)

    def train_(self, data_iter, optimizer, num_epochs=10, ls_fn: nn.Module = nn.L1Loss(),
               acc_fn=single_argmax_accuracy, valid_iter=None) -> History:
        """
        神经网络训练函数。
        :param data_iter: 训练数据供给迭代器
        :param optimizer: 网络参数优化器
        :param num_epochs: 迭代世代
        :param ls_fn: 训练损失函数
        :param acc_fn: 准确率计算函数
        :param valid_iter: 验证数据供给迭代器
        :return: 训练数据记录`History`对象
        """
        history = History('train_l', 'train_acc') if not valid_iter else \
            History('train_l', 'train_acc', 'valid_l', 'valid_acc')
        with tqdm(total=len(data_iter), unit='批', position=0,
                  desc=f'训练中...', mininterval=1) as pbar:
            for epoch in range(num_epochs):
                pbar.reset(len(data_iter))
                pbar.set_description(f'世代{epoch + 1}/{num_epochs} 训练中...')
                metric = Accumulator(3)  # 批次训练损失总和，准确率，样本数
                # 训练主循环
                for X, y in data_iter:
                    with torch.enable_grad():
                        self.train()
                        optimizer.zero_grad()
                        lo = ls_fn(self(X), y)
                        lo.backward()
                        optimizer.step()
                    with torch.no_grad():
                        correct = acc_fn(self(X), y)
                        num_examples = X.shape[0]
                        metric.add(lo.item() * num_examples, correct, num_examples)
                    pbar.update(1)
                # 记录训练数据
                if not valid_iter:
                    history.add(
                        ['train_l', 'train_acc'],
                        [metric[0] / metric[2], metric[1] / metric[2]]
                    )
                else:
                    pbar.set_description('validating...')
                    valid_acc, valid_l = self.test_(valid_iter, acc_fn, ls_fn)
                    history.add(
                        ['train_l', 'train_acc', 'valid_l', 'valid_acc'],
                        [metric[0] / metric[2], metric[1] / metric[2], valid_l, valid_acc]
                    )
            pbar.close()
        return history

    @staticmethod
    def save_grad(name):
        # 返回hook函数
        def hook(grad):
            print(f'name={name}, grad={grad}')

        return hook

    hook_mute = False

    last_forward_output = {}

    @staticmethod
    def hook_forward_fn(module, input, output):
        if not BasicNN.hook_mute:
            print(f'{module.__class__.__name__} FORWARD')
        try:
            last_input, last_output = BasicNN.last_forward_output.pop(module)
        except Exception as _:
            pass
        else:
            flag = True
            for li, i in zip(last_input, input):
                flag = torch.equal(li, i) and flag
            if not BasicNN.hook_mute:
                print(f'input eq: {flag}')
            flag = True
            for lo, o in zip(last_output, output):
                flag = torch.equal(lo, o) and flag
            if not BasicNN.hook_mute:
                print(f'output eq: {flag}')
        BasicNN.last_forward_output[module] = input, output
        # print(f'module: {module}')
        # print(f'input_eq: {input}')
        # print(f'output_eq: {output}')
        if not BasicNN.hook_mute:
            print('-' * 20)

    last_backward_data = {}

    @staticmethod
    def hook_backward_fn(module, grad_input, grad_output):
        if not BasicNN.hook_mute:
            print(f'{module.__class__.__name__} BACKWARD')
        try:
            last_input, last_output = BasicNN.last_backward_data.pop(module)
        except Exception as _:
            pass
        else:
            flag = True
            for li, i in zip(last_input, grad_input):
                if li is None or i is None:
                    print(f'{module.__class__.__name__} FORWARD None grad within {li} or {i}')
                else:
                    flag = torch.equal(li, i) and flag
                    if not BasicNN.hook_mute:
                        print(f'in_grad eq: {flag}')
            flag = True
            for lo, o in zip(last_output, grad_output):
                if lo is None or o is None:
                    print(f'None grad within {lo} or {o}')
                else:
                    flag = torch.equal(lo, o) and flag
                    if not BasicNN.hook_mute:
                        print(f'out_grad eq: {flag}')
        BasicNN.last_backward_data[module] = grad_input, grad_output
        # print(f'module: {module}')
        # print(f'grad_input: {grad_input}')
        # print(f'grad_output: {grad_output}')
        if not BasicNN.hook_mute:
            print('-' * 20)

    def train_with_hook(self, data_iter, optimizer, num_epochs=10,
                        loss: nn.Module = nn.L1Loss(),
                        acc_func=single_argmax_accuracy) -> History:
        history = History('train_l', 'train_acc')
        for m in self:
            m.register_forward_hook(hook=BasicNN.hook_forward_fn)
            m.register_full_backward_hook(hook=BasicNN.hook_backward_fn)
        for _ in trange(num_epochs, unit='epoch', desc='Epoch training...',
                        mininterval=1):
            metric = Accumulator(3)  # 批次训练损失总和，准确率，样本数
            with tqdm(total=len(data_iter), unit='batch', position=0,
                      desc=f'Epoch{_ + 1}/{num_epochs} training...',
                      mininterval=1) as pbar:
                for X, y in data_iter:
                    with torch.enable_grad():
                        self.train()
                        optimizer.zero_grad()
                        y_hat = self(X)
                        lo = loss(y_hat, y)
                        lo.backward()
                        optimizer.step()
                    with torch.no_grad():
                        correct = acc_func(y_hat, y)
                        num_examples = X.shape[0]
                        metric.add(lo.item() * num_examples, correct, num_examples)
                    pbar.update(1)
                pbar.close()
            history.add(
                ['train_l', 'train_acc'],
                [metric[0] / metric[2], metric[1] / metric[2]]
            )
        return history

    @staticmethod
    def get_k_fold_data(k, i, dataset: DataSet):
        """
        根据K、i、X、y获取训练集和验证集
        :param k: 数据集拆分折数
        :param i:
        :param X:
        :param y:
        :return:
        """
        assert k > 1
        fold_size = len(dataset) // k
        X_train, y_train = None, None
        for j in range(k):
            idx = slice(j * fold_size, (j + 1) * fold_size)
            X_part, y_part = X[idx, :], y[idx]
            if j == i:
                X_valid, y_valid = X_part, y_part
            elif X_train is None:
                X_train, y_train = X_part, y_part
            else:
                X_train = torch.cat([X_train, X_part], 0)
                y_train = torch.cat([y_train, y_part], 0)
        return X_train, y_train, X_valid, y_valid

    def train_with_k_fold(self, k, X_train, y_train, num_epochs, learning_rate, weight_decay,
           batch_size, dropout_rate, activation, netType):
        train_l_sum, valid_l_sum = 0, 0
        for i in range(k):
            data = get_k_fold_data(k, i, X_train, y_train)
            train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                       weight_decay, batch_size)
            train_l_sum += train_ls[-1]
            valid_l_sum += valid_ls[-1]
            print(f"\tfold{i} done")
        return train_l_sum / k, valid_l_sum / k

    @torch.no_grad()
    def test_(self, test_iter, acc_func=single_argmax_accuracy, loss: Callable = nn.L1Loss) \
            -> [float, float]:
        """
        测试方法，取出迭代器中的下一batch数据，进行预测后计算准确度和损失
        :param test_iter: 测试数据迭代器
        :param acc_func: 计算准确度所使用的函数
        :param loss: 计算损失所使用的函数
        :return: 测试准确率，测试损失
        """
        self.eval()
        # with torch.no_grad():
        #     for features, labels in test_iter:
        #         preds = self(features)
        #         test_acc = acc_func(preds, labels) / len(features)
        #         test_ls = loss(preds, labels)
        #         del preds
        #         return test_acc, test_ls.item()
        metric = Accumulator(3)
        for features, labels in test_iter:
            preds = self(features)
            metric.add(loss(preds, labels), acc_func(preds, labels), len(features))
        return metric[0] / metric[2], metric[1] / metric[2]

    @torch.no_grad()
    def predict_(self, feature_iter: Iterable) -> torch.Tensor:
        ret = []
        for feature in feature_iter:
            ret.append(self(feature))
        return torch.cat(ret, dim=0)

    @property
    def device(self):
        return self.__device__
