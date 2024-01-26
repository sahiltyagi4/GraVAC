import torch
import torch.distributed as dist
from torch._C._distributed_c10d import ReduceOp


class CollectiveCommOps(object):
    def __init__(self, world_size, async_op, model_stats, device):
        self.world_size = world_size
        self.async_op = async_op
        self.model_stats = model_stats
        self.device = device

    def allreduce(self, model):
        for name, parameter in model.named_parameters():
            dist.all_reduce(parameter.grad, async_op=self.async_op, op=ReduceOp.SUM)
            parameter.grad = parameter.grad / self.world_size

        return model

    def broadcast(self, model, rank=0):
        for _, param in model.named_parameters():
            if not param.requires_grad: continue
            dist.broadcast(tensor=param.data, src=rank, async_op=self.async_op)

        return model

    def layerwise_decompress(self, gathered_compgrads, gathered_ixs, i):
        p_shape = self.model_stats.param_shapes[i]
        tensor = torch.zeros(p_shape).view(-1).to(self.device)
        for ix in range(len(gathered_compgrads)):
            tensor.data[gathered_ixs[ix]] += gathered_compgrads[ix]

        tensor = tensor / self.world_size
        tensor = tensor.reshape(p_shape)
        return tensor

    def compression_allgather(self, layer_values, layer_indices):
        reduced_grads = []
        for i in range(len(layer_values)):
            comp_grad = layer_values[i]
            comp_ixs = layer_indices[i]
            tensor_sizes = [torch.LongTensor([0]).to(self.device) for _ in range(self.world_size)]
            t_size = comp_grad.numel()
            dist.all_gather(tensor_sizes, torch.LongTensor([t_size]).to(self.device))

            tensor_list = []
            ix_list = []
            size_list = [int(size.item()) for size in tensor_sizes]
            max_size = max(size_list)
            if max_size > 0:
                for _ in size_list:
                    tensor_list.append(torch.zeros(size=(max_size,), dtype=torch.float32).to(self.device))
                    ix_list.append(torch.zeros(size=(max_size,), dtype=torch.long).to(self.device))
                if t_size != max_size:
                    g_padding = torch.zeros(size=(max_size - t_size,), dtype=torch.float32).to(self.device)
                    ix_padding = torch.zeros(size=(max_size - t_size,), dtype=torch.long).to(self.device)
                    comp_grad = torch.cat((comp_grad, g_padding), dim=0).to(self.device)
                    comp_ixs = torch.cat((comp_ixs, ix_padding), dim=0).to(self.device)

                dist.all_gather(tensor_list, comp_grad)
                dist.all_gather(ix_list, comp_ixs)

                reduced_grads.append(self.layerwise_decompress(tensor_list, ix_list, i))
            else:
                reduced_grads.append(None)

        return reduced_grads