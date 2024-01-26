import torch

from gravac_py3.compressors import Compressor
from gravac_py3.compressors.residual_gradients import ResidualGrads


def sparsifier(tensor, compress_ratio, device):
    tensor = tensor.flatten()
    numel = tensor.numel()
    k = max(1, int(numel * compress_ratio))

    indices = torch.randint(low=0, high=numel-1, size=(k,)).to(device)
    values = tensor[indices].to(device)

    return values, indices

def desparsifier(tensors, numel, device):
    values, indices = tensors
    tensor_decompressed = torch.zeros(numel, dtype=values.dtype, device=device)
    tensor_decompressed.scatter_(0, indices, values).to(device)

    return tensor_decompressed


class RandomKCompressor(Compressor):
    def __init__(self, device, compress_ratio):
        super().__init__()
        self.global_step = 0
        self.device = device
        self.residual = ResidualGrads()
        self.compress_ratio = compress_ratio

    def compress(self, tensor, name):
        tensor = tensor.to(self.device)

        tensor = self.residual.compensate(tensor, name)
        numel = tensor.numel()
        shape = tensor.size()
        tensors = sparsifier(tensor, self.compress_ratio, self.device)
        ctx = numel, shape
        self.residual.update(tensor, name, self, tensors, ctx)
        return tensors, ctx

    def decompress(self, tensors, ctx):
        """Decompress by filling empty slots with zeros and reshape back using the original shape"""
        numel, shape = ctx
        tensor_decompressed = desparsifier(tensors, numel, self.device)
        return tensor_decompressed.view(shape)