import torch
from abc import ABC
import torch.nn as nn
from tqdm import tqdm
from ..decorators import *
from abc import abstractmethod
from ..constants import Training
from ..constants import ErrorMsgs
from ..logger.logger import LogModule
from ..utils.metrics import AverageMeter


class On(nn.Module, ABC):
    def __init__(self):
        nn.Module.__init__(self)
        ABC.__init__(self)
        self.loss_function = None
        self.optimizer = None
        self.lr_scheduler = None
        self.device = torch.device(Training.Defaults.GPU if torch.cuda.is_available() else Training.Defaults.DEVICE)
        self._train_avg_meter = None
        self._train_state = Training.Defaults.TRAIN_STATE
        self._test_state = Training.Defaults.TEST_STATE

    @property
    def train_state(self):
        return self._train_state

    @train_state.setter
    @decors.state_change
    def train_state(self, value):
        self._train_state = value

    @property
    def test_state(self):
        return self._test_state

    @test_state.setter
    @decors.state_change
    def test_state(self, value):
        self._test_state = value

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs):
        return

    @abstractmethod
    def train_one_step(self, batch):
        return

    @decors.record_loss
    def train_one_epoch(self, epoch, loader):
        self._train_avg_meter.reset()
        for batch in loader:

            if isinstance(batch, list):
                for index, each in enumerate(batch):
                    batch[index] = batch[index].to(self.device)
            elif isinstance(batch, dict):
                for key, value in batch.items():
                    batch[key] = batch[value].to(self.device)

            loss = self.train_one_step(batch)
            self._train_avg_meter.update(loss.cpu().detach().numpy())
        return self._train_avg_meter.avg

    def valid_one_step(self):
        ...

    def fit(self,
            train_loader,
            test_loader=None,
            epochs=Training.Defaults.EPOCHS):

        assert epochs > 0, ErrorMsgs.Training.EPOCHS_FAIL
        assert train_loader is not None, ErrorMsgs.Training.TRAIN_LOADER_NONE

        # record system usage
        # display model arx
        # send train start signal to logger
        # change train state
        # log initial hyper params

        self.train()
        self.to(device=self.device)
        self._train_avg_meter = AverageMeter()
        self.loss_function = self.configure_loss()
        self.optimizer, self.lr_scheduler = self.configure_optimizers()
        for _ in tqdm(range(epochs)):
            loss = self.train_one_epoch(_, train_loader)
            print(loss)
            # batch: object
            # for batch in train_loader:
            #     print(type(batch))
            #     loss = self.train_one_step(batch)

        # print(logits.argmax(dim=1), y)

    def predict(self, args):
        return

    def evaluate(self):
        ...

    @abstractmethod
    def configure_optimizers(self):
        return

    @abstractmethod
    def configure_loss(self):
        return

    def optimization_step(self):
        self.optimizer.step()
        # self.scheduler.step()

    def on_train_start(self):
        ...

    def on_epoch_end(self):
        self.scheduler.step()
        ...

    def on_train_end(self):
        ...

    def on_valid_end(self):
        ...

    def on_test_end(self):
        ...

    def save_model(self):
        ...

    def load_model(self):
        ...
