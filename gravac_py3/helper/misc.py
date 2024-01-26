import numpy as np
import random
from _random import Random
import pandas as pd
from prettytable import PrettyTable
from datetime import datetime

import torch

from gravac_py3.helper.dataloader import TestData
from gravac_py3.models.image_classifiers import Resnet101, VGG16, LSTMmodel
from gravac_py3.compressors.topk_compression import TopKCompressor
from gravac_py3.compressors.dgc_compression import DgcCompressor
from gravac_py3.compressors.redsync_compression import RedsyncCompressor
from gravac_py3.compressors.randomk_compression import RandomKCompressor
from gravac_py3.compressors.variable_compression import VariableTopKCompressor

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    rng = Random()
    rng.seed(seed)
    torch.use_deterministic_algorithms(False)


def sync_device(device):
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)

    return


def get_model(model_name, args, batch_update_size):

    if model_name == 'resnet101':
        model = Resnet101(args.lr, args.momentum, args.weight_decay, batch_update_size, args.seed)
    elif model_name == 'vgg16':
        model = VGG16(args.lr, args.momentum, args.weight_decay, batch_update_size, args.seed)
    elif model_name == 'lstm':
        _, vocab_size = TestData(args).test_data()
        model = LSTMmodel(args.train_bsz, vocab_size, args.lr, args.momentum, args.weight_decay, batch_update_size,
                          args.seed)

    return model


def test_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def timed_log(text):
    print(f"{datetime.now().strftime('%H:%M:%S')} {text}")


class MovingWindowAverage(object):
    def __init__(self, window_size, alpha):
        self.reset(window_size)
        self.alpha = alpha

    def reset(self, window_size):
        self.windowlist = []
        self.meanval = 0.0
        self.windowsize = window_size

    def compute_moving_avg(self, val):
        self.windowlist.append(val)
        if len(self.windowlist) == self.windowsize:
            df = pd.DataFrame(self.windowlist)
            ewm_vals = df.ewm(alpha=self.alpha, min_periods=self.windowsize).mean().values.tolist()
            self.meanval = ewm_vals[-1][0]
            self.windowlist.pop(0)

        return self.meanval


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ModelHelper(object):
    def __init__(self, model, compressor=None):
        self.model = model
        self.compressor = compressor
        self.param_names = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.param_names.append(name)

    def count_model_parameters(self):
        table = PrettyTable(["Module", "Parameters"])
        total_params = 0
        ctr = 0
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad: continue
            param = parameter.numel()
            table.add_row([name, param])
            total_params += param
            ctr += 1

        print(table)
        print(f"counter is {ctr}")
        print(f"Total Trainable Params: {total_params}")
        total_size = (total_params * 4) / (1024 * 1024)
        print(f"Model memory footprint using single precision: {total_size} MB")
        return total_params

    def grad_norm(self, grads):
        gnorm = 0.0
        for g in grads:
            layer_grad_norm = torch.norm(g.flatten())
            gnorm += layer_grad_norm

        return gnorm

    def gradient_compression(self, gradients, compress_ratio):
        with torch.no_grad():
            layer_values, layer_indices = [], []
            for ix in range(len(gradients)):
                tensor_compressed, _ = self.compressor.compress(gradients[ix], self.param_names[ix], compress_ratio)
                layer_values.append(tensor_compressed[0])
                layer_indices.append(tensor_compressed[1])

            return layer_values, layer_indices

    def effective_ratio(self, tensors_uncompressed, tensors_compressed, device):
        with torch.no_grad():
            uncompressed_sum = 0
            compressed_sum = 0
            for ix in range(len(tensors_uncompressed)):
                uncompressed_sum += tensors_uncompressed[ix].to(device).numel()
                compressed_sum += tensors_compressed[ix].to(device).numel()

            return float(compressed_sum / uncompressed_sum)

    def compression_gain(self, uncompressed_norm, compressed_norm):

        return float(compressed_norm / uncompressed_norm)

    def layerwise_compressed_grads(self):
        with torch.no_grad():
            layerwise_compressed_tensors = []
            for ix in range(len(self.param_names)):
                layerwise_compressed_tensors.append(self.compressor.residual.layer_decompress[self.param_names[ix]])

            return layerwise_compressed_tensors


class CompressionType(object):
    def __init__(self, compression, device, compress_ratio=1.0):
        self.compression = compression
        self.device = device
        self.compress_ratio = compress_ratio

    def get_compressor(self):
        compressor = None
        if self.compression == 'topK':
            compressor = TopKCompressor(device=self.device, compress_ratio=self.compress_ratio)
        elif self.compression == 'dgc':
            compressor = DgcCompressor(device=self.device, compress_ratio=self.compress_ratio)
        elif self.compression == 'redsync':
            compressor = RedsyncCompressor(device=self.device, compress_ratio=self.compress_ratio)
        elif self.compression == 'randomK':
            compressor = RandomKCompressor(device=self.device, compress_ratio=self.compress_ratio)
        elif self.compression == 'gravacTopK':
            compressor = VariableTopKCompressor(device=self.device)

        return compressor