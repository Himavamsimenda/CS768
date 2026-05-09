import torch
import torch.nn as nn


class Norm(nn.Module):
    def __init__(self, norm_type, hidden_dim=64, print_info=None):
        super(Norm, self).__init__()
        self.norm = None
        self.print_info = print_info

        if norm_type == 'bn':
            self.norm = nn.BatchNorm1d(hidden_dim)

        elif norm_type in ['gn', 'gsgn', 'dwgn']:
            # gn   = original GraphNorm
            # gsgn = graph-size-aware GraphNorm
            # dwgn = degree-weighted GraphNorm
            self.norm = norm_type
            self.weight = nn.Parameter(torch.ones(hidden_dim))
            self.bias = nn.Parameter(torch.zeros(hidden_dim))
            self.mean_scale = nn.Parameter(torch.ones(hidden_dim))

            if norm_type == 'gsgn':
                self.size_gate = nn.Sequential(
                    nn.Linear(1, hidden_dim),
                    nn.Sigmoid()
                )

        elif norm_type is None:
            self.norm = None

        else:
            raise ValueError(f"Unsupported norm_type: {norm_type}")

    def forward(self, graph, tensor, print_=False):
        # BatchNorm path
        if self.norm is not None and type(self.norm) != str:
            return self.norm(tensor)

        # No norm path
        if self.norm is None:
            return tensor

        batch_list = graph.batch_num_nodes()
        batch_size = len(batch_list)
        batch_list = batch_list.clone().detach().to(device=tensor.device, dtype=torch.long)

        batch_index = torch.arange(batch_size, device=tensor.device).repeat_interleave(batch_list)
        batch_index_expand = batch_index.view((-1,) + (1,) * (tensor.dim() - 1)).expand_as(tensor)

        if self.norm == 'dwgn':
            # ----- Degree-Weighted GraphNorm -----
            deg = graph.in_degrees().float().to(tensor.device)

            # use a smooth positive weight; avoids huge domination by hubs
            node_weight = 1.0 + torch.log1p(deg)   # shape [N]
            node_weight_expand = node_weight.view(-1, 1).expand_as(tensor)

            # sum of weights per graph
            weight_sum = torch.zeros(batch_size, device=tensor.device)
            weight_sum = weight_sum.scatter_add_(0, batch_index, node_weight)
            weight_sum = weight_sum.clamp(min=1e-6)

            # weighted graph mean
            mean = torch.zeros(batch_size, *tensor.shape[1:], device=tensor.device)
            mean = mean.scatter_add_(0, batch_index_expand, tensor * node_weight_expand)
            mean = mean / weight_sum.unsqueeze(-1)
            mean_per_node = mean.repeat_interleave(batch_list, dim=0)

            alpha = self.mean_scale.unsqueeze(0).expand(batch_size, -1)
            alpha_per_node = alpha.repeat_interleave(batch_list, dim=0)

            sub = tensor - mean_per_node * alpha_per_node

            # weighted graph std
            var = torch.zeros(batch_size, *tensor.shape[1:], device=tensor.device)
            var = var.scatter_add_(0, batch_index_expand, sub.pow(2) * node_weight_expand)
            var = var / weight_sum.unsqueeze(-1)
            std = (var + 1e-6).sqrt()
            std = std.repeat_interleave(batch_list, dim=0)

            return self.weight * sub / std + self.bias

        # ----- Original GN / your GSGN path -----
        mean = torch.zeros(batch_size, *tensor.shape[1:], device=tensor.device)
        mean = mean.scatter_add_(0, batch_index_expand, tensor)
        mean = (mean.T / batch_list).T
        mean_per_node = mean.repeat_interleave(batch_list, dim=0)

        if self.norm == 'gn':
            alpha = self.mean_scale.unsqueeze(0).expand(batch_size, -1)

        elif self.norm == 'gsgn':
            graph_size = torch.log1p(batch_list.float()).unsqueeze(-1)
            size_gate = self.size_gate(graph_size)
            alpha = self.mean_scale.unsqueeze(0) * size_gate

        else:
            raise ValueError(f"Unexpected norm mode: {self.norm}")

        alpha_per_node = alpha.repeat_interleave(batch_list, dim=0)

        sub = tensor - mean_per_node * alpha_per_node

        std = torch.zeros(batch_size, *tensor.shape[1:], device=tensor.device)
        std = std.scatter_add_(0, batch_index_expand, sub.pow(2))
        std = ((std.T / batch_list).T + 1e-6).sqrt()
        std = std.repeat_interleave(batch_list, dim=0)

        return self.weight * sub / std + self.bias