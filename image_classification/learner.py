# zhangshulin
# zhangslwork@yeah.net
# 2019-10-14


import torch
import torch.nn as nn
import torch.optim as optim


class Learner:
    def __init__(self, model):
        if torch.cuda.is_available():
            self.model = model.to('cuda')
        else:
            self.model = model

    def fit(self, dataloader, lr, epochs, weight_decay=0, print_steps=200):
        self.model.train()

        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr, momentum=0.9,
                              weight_decay=weight_decay, nesterov=True)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                            mode='min',
                                                            verbose=True)

        for epoch in range(epochs):
            for step, (imgs, labels) in enumerate(dataloader):
                if torch.cuda.is_available():
                    imgs, labels = imgs.to('cuda'), labels.to('cuda')

                outputs = self.model(imgs)
                loss = loss_fn(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                lr_scheduler.step(loss)

                if step % print_steps == print_steps - 1:
                    print(f"epoch: {epoch + 1}    \tstep: {step + 1}    \tloss: {loss:.4f}")

    def evaluate(self, dataloader):
        self.model.eval()

        total = 0
        corrects = 0

        with torch.no_grad():
            for imgs, labels in dataloader:
                if torch.cuda.is_available():
                    imgs, labels = imgs.to('cuda'), labels.to('cuda')

                outputs = self.model(imgs)
                predictions = outputs.argmax(dim=1)
                corrects += (predictions == labels).sum().item()
                total += imgs.size(0)

        return corrects / total


