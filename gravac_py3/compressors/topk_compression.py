import torch

from gravac_py3.compressors import Compressor
from gravac_py3.compressors.residual_gradients import ResidualGrads


def sparsifier(tensor, compress_ratio):
    tensor = tensor.flatten()
    k = max(1, int(tensor.numel() * compress_ratio))

    _, indices = torch.topk(tensor.abs(), k, sorted=False,)
    values = torch.gather(tensor, 0, indices)

    return values, indices
    # k = max(1, int(tensor.numel() * compress_ratio))
    # values, indexes = tensor.abs().sort(descending=True)
    #
    # return values[:k], indexes[:k]


def desparsifier(tensors, numel, device):
    values, indices = tensors
    tensor_decompressed = torch.zeros(numel, dtype=values.dtype, device=device)
    tensor_decompressed.scatter_(0, indices, values)

    return tensor_decompressed


class TopKCompressor(Compressor):

    def __init__(self, device, compress_ratio):
        super().__init__()
        self.residual = ResidualGrads()
        self.device = device
        self.compress_ratio = compress_ratio

    def compress(self, tensor, name):
        tensor = tensor.to(self.device)

        tensor = self.residual.compensate(tensor, name)
        numel = tensor.numel()
        shape = tensor.size()
        tensors = sparsifier(tensor, self.compress_ratio)
        ctx = numel, shape
        self.residual.update(tensor, name, self, tensors, ctx)

        return tensors, ctx

    def decompress(self, tensors, ctx):
        """Decompress by filling empty slots with zeros and reshape back using the original shape"""
        numel, shape = ctx
        tensor_decompressed = desparsifier(tensors, numel, self.device)

        return tensor_decompressed.view(shape)