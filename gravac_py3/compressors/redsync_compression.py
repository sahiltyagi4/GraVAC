import torch

from gravac_py3.compressors import Compressor
from gravac_py3.compressors.residual_gradients import ResidualGrads


class RedsyncCompressor(Compressor):

    def __init__(self, device, compress_ratio):
        super().__init__(tensors_size_are_same=False)
        self.device = device
        self.residual = ResidualGrads()
        self.compress_ratio = compress_ratio

    def compress(self, tensor, name):
        tensor = tensor.to(self.device)

        tensor = self.residual.compensate(tensor, name)
        shape = tensor.size()
        tensor = tensor.flatten()
        numel = tensor.numel()
        k = max(int(numel * self.compress_ratio), 1)

        l = 0.0
        r = 1.0
        thres = 0.0
        eps = 0.2
        abs_tensor = torch.abs(tensor)
        mean_val = torch.mean(abs_tensor)
        max_val = torch.max(abs_tensor)

        while r - l > eps:
            tmp_ratio = l + (r - l) / 2
            thres = mean_val + tmp_ratio * (max_val - mean_val)
            one_indexes = (abs_tensor > thres)
            indexes = one_indexes.nonzero().data.squeeze().view(-1)
            nnz = indexes.numel()
            if nnz > k and 2 * k > nnz:
                break
            elif nnz < k / 2:
                r = tmp_ratio
            else:
                l = tmp_ratio

        values = tensor.data[indexes]
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