import torch

from gravac_py3.compressors import VariableCompressor
from gravac_py3.compressors.residual_gradients import ResidualGrads


# currently implements top-k compression where compression ratio is supplied as argument to compress fn.
# the idea is to be able to use different compression ratios over the iterations


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


class VariableTopKCompressor(VariableCompressor):

    def __init__(self, device):
        super().__init__()
        self.residual = ResidualGrads()
        self.device = device

    def compress(self, tensor, name, compress_ratio):
        tensor = tensor.to(self.device)

        tensor = self.residual.compensate(tensor, name)
        numel = tensor.numel()
        shape = tensor.size()
        tensors = sparsifier(tensor, compress_ratio)
        ctx = numel, shape
        self.residual.update(tensor, name, self, tensors, ctx)

        return tensors, ctx

    def decompress(self, tensors, ctx):
        """Decompress by filling empty slots with zeros and reshape back using the original shape"""
        numel, shape = ctx
        tensor_decompressed = desparsifier(tensors, numel, self.device)

        return tensor_decompressed.view(shape)