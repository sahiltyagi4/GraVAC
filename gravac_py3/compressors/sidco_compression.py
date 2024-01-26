import math

import torch

from gravac_py3.compressors import Compressor
from gravac_py3.compressors.residual_gradients import ResidualGrads


# modeling gradients as a double-exponential distribution
class SidcoExp(Compressor):

    def __init__(self, num_stages, device, compress_ratio):
        super().__init__(tensors_size_are_same=False)
        self.num_stages = num_stages
        self.device = device
        self.compress_ratio = compress_ratio
        self.residual = ResidualGrads()
        self.first_ratio = 0.75
        self.i_ratio = 0.25

    def compress(self, tensor, name):
        tensor = tensor.to(self.device)
        tensor = self.residual.compensate(tensor, name)
        shape = tensor.size()
        tensor = tensor.flatten()
        numel = tensor.numel()
        t_norm = tensor.norm(2)
        abs_norm_tensor = (tensor.abs() / t_norm)
        abs_norm_tensor_cpy = abs_norm_tensor.clone()
        t_mean = torch.mean(abs_norm_tensor)
        if self.num_stages == 1 or self.compress_ratio >= self.first_ratio:
            threshold = -t_mean * math.log(self.compress_ratio)
        else:
            threshold = -t_mean * math.log(self.first_ratio)

        r_ratio = self.compress_ratio / self.first_ratio
        if self.num_stages > 1 or self.num_stages == 0:
            if self.num_stages == 0:
                loop = (math.ceil(math.log(r_ratio) / math.log(self.i_ratio)))
            else:
                self.i_ratio = (math.pow(r_ratio, 1.0 / (self.num_stages - 1)))
                loop = self.num_stages - 1
            i = loop
            while i > 0:
                one_indexes = (abs_norm_tensor > threshold).to(self.device)
                indexes = one_indexes.nonzero().data.squeeze().view(-1).to(self.device)
                abs_norm_tensor = abs_norm_tensor.data[indexes].to(self.device)

                # to handle failure when # stages renders abs_norm_tensor to be empty
                if abs_norm_tensor.size()[0] > 0:
                    t_min = abs_norm_tensor.min()
                    t_mean = torch.mean(abs_norm_tensor)

                    threshold = (-(t_mean - t_min) * math.log(self.i_ratio) + t_min)
                    if i == 1 and self.num_stages == 0:
                        threshold = (-(t_mean - t_min) * math.log(r_ratio / math.pow(self.i_ratio, loop - 1)) + t_min)
                    i -= 1
                else:
                    break

        one_indexes = (abs_norm_tensor_cpy > threshold).to(self.device)
        indexes = one_indexes.nonzero().data.squeeze().view(-1).to(self.device)
        values = tensor.data[indexes].to(self.device)
        tensors = values, indexes
        ctx = numel, shape
        self.residual.update(tensor.view(shape), name, self, tensors, ctx)
        return tensors, ctx

    def decompress(self, tensors, ctx):
        """Decompress by filling empty slots with zeros and reshape back using the original shape"""
        numel, shape = ctx
        values, indices = tensors
        tensor_decompressed = torch.zeros(numel, dtype=values.dtype, device=self.device)
        tensor_decompressed.scatter_(0, indices, values).to(self.device)
        return tensor_decompressed.view(shape)