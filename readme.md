BrightLite - Brighter and Lite than Torch

Bright-Lite is an early stage dev version a Deep Learning Framework based on PyTorch. I  started this as a personal project speed up the dev process so that I can invest more time on something else.

### Example using bright-lite

```
import torch
import brightlite
import torchvision
import torch.nn as nn


train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./datasets/', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=8, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./datasets/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=8, shuffle=True)


class MyMNISTModel(brightlite.On):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        return self.linear(x)


class LowBeam(brightlite.Beam):
    def __init__(self, _model):
        super().__init__(_model)

    def train_one_step(self, batch):
        X, y = batch
        logits = self.model(x=X.view(8, 784))
        loss = self.loss_function(logits, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimization_step()
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        lr_scheduler = None
        return optimizer, lr_scheduler

    def configure_loss(self):
        return torch.nn.CrossEntropyLoss()

    def predict(self, test_loader):
        for b in test_loader:
            X, y = b
            y_pred = self.model(x=X.view(8, 784))
            print(y, torch.argmax(y_pred, dim=1))


model = MyMNISTModel()
beam = LowBeam(model)
beam.fit(train_loader, test_loader, epochs=5, multiple_gpus=True)
beam.predict(test_loader)

```
