import torch

from gravac_py3.compressors import Compressor
from gravac_py3.compressors.residual_gradients import ResidualGrads


class DgcCompressor(Compressor):

    def __init__(self, device, compress_ratio):
        super().__init__(tensors_size_are_same=False)
        self.residual = ResidualGrads()
        self.device = device
        self.compress_ratio = compress_ratio

    def compress(self, tensor, name):
        tensor = tensor.to(self.device)

        tensor = self.residual.compensate(tensor, name)
        shape = tensor.size()
        tensor = tensor.flatten()
        numel = tensor.numel()

        sample_shape = [max(1, int(numel * 0.01))]
        sample_index = torch.empty(sample_shape).uniform_(0, numel).type(torch.long)
        sample_tensor = tensor[sample_index]

        # k = max(1, int(numel * compress_ratio * 0.01))
        # vals, indices = torch.topk(sample_tensor.abs(), k)

        k = max(1, int(numel * self.compress_ratio * 0.01))
        vals, indices = sample_tensor.abs().sort(descending=True)
        vals, indices = vals[:k], indices[:k]

        thr = vals.min()
        mask = (tensor.abs() >= thr)
        selected = mask.sum()

        for _ in range(10):
            if selected > 1.3 * numel * self.compress_ratio:
                thr = 1.3 * thr
            elif selected < 0.7 * numel * self.compress_ratio:
                thr = 0.7 * thr
            else:
                break

            mask = (tensor.abs() >= thr)
            selected = mask.sum()

        indices, = torch.where(mask)
        values = tensor[indices]

        tensor_compressed = values, indices
        ctx = numel, shape
        self.residual.update(tensor.view(shape), name, self, tensor_compressed, ctx)

        return tensor_compressed, ctx

    def decompress(self, tensor_compressed, ctx):
        values, indices = tensor_compressed
        numel, shape = ctx
        tensor_decompressed = torch.zeros(numel, dtype=values.dtype, device=self.device)
        tensor_decompressed.scatter_(0, indices, values)

        return tensor_decompressed.view(shape)