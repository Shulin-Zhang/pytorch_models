# zhangshulin
# zhangslwork@yeah.net
# 2019-10-19


import copy
import torch
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt
import matplotlib


class Lr_finder:
    def __init__(self, model, dataloader, loss_fn, optimizer):
        if torch.cuda.is_available():
            self.model = model.to('cuda')
        else:
            self.model = model

        self.dataloader = dataloader
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def lr_find(self, steps=150, repeat=5, lr_range=(1e-4, 10), plot=True):
        self.model.train()

        old_state = copy.deepcopy(self.model.state_dict())

        history_loss = []
        history_lr = []

        for i in range(repeat):
            step = 0
            losses = []
            lrs = []
            stop_flag = False
            while True:
                for datas, labels in self.dataloader:
                    if torch.cuda.is_available():
                        datas, labels = datas.to('cuda'), labels.to('cuda')

                    lr = self.exp_lr_scheduler(step, steps, lr_range)
                    self.optimizer.param_groups[0]['lr'] = lr

                    outputs = self.model(datas)
                    loss = self.loss_fn(outputs, labels)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    losses.append(loss.item())
                    lrs.append(lr)

                    step += 1
                    if step == steps:
                        stop_flag = True
                        break

                if stop_flag:
                    break

            self.model.load_state_dict(old_state)

            history_loss.append(losses)
            history_lr.append(lrs)

        x = np.array(history_lr).mean(axis=0)
        y = np.array(history_loss).mean(axis=0)

        if plot:
            plt.xscale('log')
            plt.xlabel('lr')
            plt.ylabel('loss')
            plt.title('loss_lr curve')
            plt.plot(x, y)
        else:
            return x, y

    def exp_lr_scheduler(self, step, steps, lr_range):
        pct = step / (steps - 1)
        return lr_range[0] * (lr_range[1] / lr_range[0]) ** pct
