import sys
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


# beam controls the intensity
class Beam(ABC):
    def __init__(self, model):
        """

        :type model: nn.Module

        """
        ABC.__init__(self)
        self.model = model
        self.loss_function = None
        self.optimizer = None
        self.lr_scheduler = None
        self.device = torch.device(Training.Defaults.CUDA if torch.cuda.is_available() else Training.Defaults.DEVICE)
        self._train_avg_meter = None
        self._test_avg_meter = None
        self._train_state = Training.Defaults.TRAIN_STATE
        self._validate_state = Training.Defaults.VALID_STATE

    @property
    def train_state(self):
        return self._train_state

    @train_state.setter
    @decors.state_change
    def train_state(self, value):
        self._train_state = value

    @property
    def validate_state(self):
        return self._validate_state

    @validate_state.setter
    @decors.state_change
    def validate_state(self, value):
        self._validate_state = value

    @abstractmethod
    def train_one_step(self, batch):
        return

    def validate_one_step(self, batch):
        return

    def predict_one_step(self, batch):
        return

    def on_train_start(self):
        return

    def on_valid_start(self):
        return

    def on_epoch_end(self):
        # Todo
        # 1. initialize test hook
        if self.lr_scheduler:
            self.lr_scheduler.step()
        
    def on_train_end(self):
        return

    def on_valid_end(self):
        return

    def _to_device(self, batch):
        if isinstance(batch, list):
            for index, _ in enumerate(batch):
                    batch[index] = batch[index].to(self.device)
        elif isinstance(batch, dict):
            for key, value in batch.items():
                batch[key] = batch[value].to(self.device)
        return batch

    @decors.record_loss
    def train_one_epoch(self, epoch, loader):
        self._train_avg_meter.reset()

        for batch in loader:
            batch = self._to_device(batch)
            loss = self.train_one_step(batch)
            self._train_avg_meter.update(loss.cpu().detach().numpy())

        return self._train_avg_meter.avg
    

    def fit(self,
            train_loader,
            test_loader=None,
            epochs=Training.Defaults.EPOCHS,
            callbacks=None,
            multiple_gpus=True):

        assert epochs >= Training.Defaults.EPOCHS, ErrorMsgs.Training.EPOCHS_FAIL
        assert train_loader is not None, ErrorMsgs.Training.TRAIN_LOADER_NONE

        # changing training status
        self._train_state = Training.WarmUp.TRAIN_STATE

        # record system usage
        # display model arx
        # send train start signal to logger
        # change train state
        # log initial hyper params
        # LogModule.logger.info(str(self))
        

        if multiple_gpus and torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)

        self.model.train()
        self.model.to(device=self.device)
        self._train_avg_meter = AverageMeter()
        self.optimizer, self.lr_scheduler = self.configure_optimizers()
        try:
            for epoch in tqdm(range(epochs), file=sys.stdout):
                self.train_state = Training.Active.TRAIN_STATE
                loss = self.train_one_epoch(epoch, train_loader)
                # callback to tensorboard
                self.train_state = Training.Pause.TRAIN_STATE
                self.validate_state = Training.WarmUp.VALID_STATE
                self.validate(test_loader)
                self.validate_state = Training.Pause.VALID_STATE
            self.train_state = Training.Defaults.TRAIN_STATE
            self.validate_state = Training.Defaults.VALID_STATE
        except KeyboardInterrupt:
            self.train_state = Training.Defaults.TRAIN_STATE
            LogModule.logger.warning("KeyboardInterrupt")

    @abstractmethod
    def configure_optimizers(self):
        return

    def optimization_step(self):
        self.optimizer.step()

    def predict(self, loader):
        self.model.eval()
        for batch in loader:
            batch = self._to_device(batch)
            with torch.no_grad():        
                self.predict_one_step(batch)

    @decors.record_metrics
    def validate(self, dataloader):
        self.model.eval()
        self._test_avg_meter = AverageMeter()
        
        for batch in dataloader:
            batch = self._to_device(batch)
            with torch.no_grad():  
                loss, metrics = self.validate_one_step(batch)
                # send loss and accuracy to callbacks
                self._test_avg_meter.update(loss)
        return self._test_avg_meter.avg, metrics

    def save_model(self):
        return

    def load_model(self):
        return

