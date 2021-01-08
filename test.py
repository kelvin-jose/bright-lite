import torch
import brightlite
import torchvision
import torch.nn as nn

from sklearn.metrics import accuracy_score


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
        loss = torch.nn.CrossEntropyLoss()(logits, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimization_step()
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        lr_scheduler = None
        return optimizer, lr_scheduler
    
    def validate_one_step(self, batch):
        X, y = batch
        y_pred = self.model(x=X.view(8, 784))
        # calculate metric
        loss = torch.nn.CrossEntropyLoss()(y_pred, y)
        acc = accuracy_score(y.cpu().detach().numpy(), torch.argmax(y_pred, dim=1).cpu().detach().numpy())
        metrics = {"accuracy": acc}
        return loss.cpu().detach().numpy(), metrics

    def predict_one_step(self, batch):
        X, y = batch
        y_pred = self.model(x=X.view(8, 784))
        # print(y.cpu().detach().numpy(), torch.argmax(y_pred, dim=1).cpu().detach().numpy())


model = MyMNISTModel()
beam = LowBeam(model)
beam.fit(train_loader, test_loader, epochs=1, multiple_gpus=True)
beam.predict(test_loader)
