import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.init as init


class SparseMM(torch.autograd.Function):
    """
    Legacy autograd function with non-static forward method is deprecated
    Implementation adapted from https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function
    """

    @staticmethod
    def forward(ctx, sparse, dense):
        output = torch.bmm(sparse, dense)
        ctx.save_for_backward(sparse)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (sparse,) = ctx.saved_tensors
        grad_input = None
        sparse_t = []
        for item in sparse:
            sparse_t.append(torch.unsqueeze(item.t(), dim=0))
        sparse_t = torch.cat(sparse_t, dim=0)
        if ctx.needs_input_grad[0]:
            grad_input = torch.bmm(sparse_t, grad_output)
        return None, grad_input


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=None):
        """Graph Convolutional Network
        Args:
            :param in_features: expected dimensionality for input to GCN
            :param out_features: dimensionality of output of GCN
            :param bias: to add a bias term to the network or not
        Methods:
            public:
                forward(inputs, adj, max_len) ---> GCN processed output
                    Forward pass over the computational graph
                reset_parameters() ---> None
                    Intializes the parameters of the GCN
        """

        # Constructor
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        init.xavier_uniform(self.weight.data, gain=1)
        self.weight.data.uniform_(-stdv, stdv)  # random
        if self.bias is not None:
            init.xavier_uniform(self.bias.data, gain=1)
            self.bias.data.uniform_(-stdv, stdv)  # random

    def forward(self, inputs, adj, max_len):
        weight_matrix = self.weight.repeat(inputs.shape[0], 1, 1).cuda()
        support = torch.bmm(inputs, weight_matrix).cuda()
        adj = adj.to("cuda")
        output = SparseMM.apply(adj, support)
        output = output * math.sqrt(max_len)
        if self.bias is not None:
            return output + self.bias.repeat(output.size(0))
        else:
            return output

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )
