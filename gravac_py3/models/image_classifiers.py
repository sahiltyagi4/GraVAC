import torch
import torch.nn as nn
from torch import optim
import torchvision.models as models

from gravac_py3.helper import misc
from gravac_py3.models import lstm_model


class Resnet101(object):
    def __init__(self, lr, momentum, weight_decay, batch_update_size, seed):
        misc.set_seed(seed)
        self.model = models.resnet101(pretrained=False, progress=True)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.batch_update_size = batch_update_size
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum,
                                   weight_decay=self.weight_decay)

    def get_model(self):
        return self.model

    def get_optimizer(self):
        return self.optimizer

    def get_lrschedule(self):
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                 milestones=[100 // self.batch_update_size,
                                                                             150 // self.batch_update_size],
                                                                 gamma=0.1, last_epoch=-1)
        return self.lr_scheduler

    def get_loss(self):
        return nn.CrossEntropyLoss()


class VGG16(object):
    def __init__(self, lr, momentum, weight_decay, batch_update_size, seed):
        misc.set_seed(seed)
        self.model = models.vgg16_bn(pretrained=False, progress=True)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.batch_update_size = batch_update_size
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum,
                                   weight_decay=self.weight_decay)

    def get_model(self):
        return self.model

    def get_optimizer(self):
        return self.optimizer

    def get_lrschedule(self):
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                 milestones=[60 // self.batch_update_size,
                                                                             120 // self.batch_update_size,
                                                                             160 // self.batch_update_size],
                                                                 gamma=0.2, last_epoch=-1)
        return self.lr_scheduler

    def get_loss(self):
        return nn.CrossEntropyLoss()


class LSTMmodel(object):
    def __init__(self, batch_size, vocab_size, lr, momentum, weight_decay, batch_update_size, seed):
        misc.set_seed(seed)
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.batch_update_size = batch_update_size
        self.model = lstm_model.LSTM(vocab_size=self.vocab_size, batch_size=self.batch_size)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                                   momentum=self.momentum, nesterov=False)

    def get_model(self):
        return self.model

    def get_optimizer(self):
        return self.optimizer

    def get_lrschedule(self):
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                 milestones=[20 // self.batch_update_size,
                                                                             32 // self.batch_update_size,
                                                                             40 // self.batch_update_size],
                                                                 gamma=0.1, last_epoch=-1)
        return self.lr_scheduler

    def get_loss(self):
        return nn.CrossEntropyLoss()