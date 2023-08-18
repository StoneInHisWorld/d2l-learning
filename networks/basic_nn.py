import torch
import torch.nn as nn
from torch.nn import Module
from tqdm import tqdm, trange

from utils.accumulator import Accumulator
from utils.data_related import single_argmax_accuracy
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

    def train_(self, data_iter, optimizer, num_epochs=10, loss: nn.Module = nn.L1Loss(),
               acc_func=single_argmax_accuracy, valid_iter=None) -> History:
        history = History('train_l', 'train_acc') if not valid_iter else \
            History('train_l', 'train_acc', 'valid_l', 'valid_acc')
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
                        lo = loss(self(X), y)
                        lo.backward()
                        optimizer.step()
                    with torch.no_grad():
                        correct = acc_func(self(X), y)
                        num_examples = X.shape[0]
                        metric.add(lo.item() * num_examples, correct, num_examples)
                    pbar.update(1)
                pbar.close()
            if not valid_iter:
                history.add(
                    ['train_l', 'train_acc'],
                    [metric[0] / metric[2], metric[1] / metric[2]]
                )
            else:
                valid_acc, valid_l = self.test_(valid_iter, acc_func, loss)
                history.add(
                    ['train_l', 'train_acc', 'valid_l', 'valid_acc'],
                    [metric[0] / metric[2], metric[1] / metric[2], valid_l, valid_acc]
                )
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

    def test_(self, test_iter, acc_func=single_argmax_accuracy, loss=nn.L1Loss) \
            -> [float, float]:
        """
        测试方法，取出迭代器中的下一batch数据，进行预测后计算准确度和损失
        :param test_iter: 测试数据迭代器
        :param acc_func: 计算准确度所使用的函数
        :param loss: 计算损失所使用的函数
        :return: 测试准确率，测试损失
        """
        with torch.no_grad():
            self.eval()
            for features, labels in test_iter:
                preds = self(features)
                test_acc = acc_func(preds, labels) / len(features)
                test_ls = loss(preds, labels)
                return test_acc, test_ls.item()

    @property
    def device(self):
        return self.__device__
