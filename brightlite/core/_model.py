from abc import ABC
import torch.nn as nn
from abc import abstractmethod


class On(nn.Module, ABC):
    def __init__(self):
        nn.Module.__init__(self)
        ABC.__init__(self)

    @abstractmethod
    def forward(self, *args, **kwargs):
        return NotImplementedError
