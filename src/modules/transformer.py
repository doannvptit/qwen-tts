import math
import torch
from torch import nn
import torch.nn.functional as F

from .conv import Conv1d


class EncoderLayers(nn.Module):
    def __init__(
        self,
        num_layers,
        d_model,
        ffn_hidden,
        n_head,
        kernel_size,
        drop_prob,
        causal=False,
        max_position=1024,
    ):
        super(EncoderLayers, self).__init__()
        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    d_model,
                    ffn_hidden,
                    n_head,
                    kernel_size,
                    drop_prob,
                    causal=causal,
                    max_position=max_position,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, src_mask, clear_mask=None):
        for layer in self.layers:
            x = layer(x, src_mask, clear_mask=clear_mask)
        return x


class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        ffn_hidden,
        n_head,
        kernel_size,
        drop_prob,
        causal=False,
        max_position=1024,
    ):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(
            d_model=d_model,
            d_hidden=ffn_hidden,
            kernel_size=kernel_size,
            dropout=drop_prob,
            causal=causal,
        )
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)
        if causal:
            self.causal_mask = torch.tril(torch.ones(max_position, max_position)) < 0.5
        else:
            self.causal_mask = None

    def forward(self, x, src_mask, clear_mask=None):
        # 0. update clear_mask if it is causal
        if self.causal_mask is not None:
            if self.causal_mask.device != x.device:
                self.causal_mask = self.causal_mask.to(x.device)

            if clear_mask is None:
                clear_mask = self.causal_mask[: x.size(1), : x.size(1)]
            else:
                clear_mask = clear_mask & self.causal_mask[: x.size(1), : x.size(1)]

        # 1. compute self attention
        _x = x
        x = self.attention.forward(q=x, k=x, v=x, mask=src_mask, clear_mask=clear_mask)

        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        # 3. positionwise feed forward network
        _x = x
        x = self.ffn(x)

        # 4. add and norm
        x = self.dropout2(x)
        x = self.norm2(x + _x)

        # 5. clear with mask
        x = x.masked_fill(src_mask.unsqueeze(-1), 0)
        return x


class PositionwiseFeedForward(nn.Module):
    """A two-feed-forward-layer module"""

    def __init__(self, d_model, d_hidden, kernel_size, dropout=0.1, causal=False):
        super().__init__()

        # Use Conv1D
        # position-wise
        self.w_1 = Conv1d(
            d_model,
            d_hidden,
            kernel_size=kernel_size[0],
            padding=(kernel_size[0] - 1) // 2,
            causal=causal,
            auto_transpose=True,
        )
        # position-wise
        self.w_2 = Conv1d(
            d_hidden,
            d_model,
            kernel_size=kernel_size[1],
            padding=(kernel_size[1] - 1) // 2,
            causal=causal,
            auto_transpose=True,
        )

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = self.w_2(F.relu(self.w_1(x)))
        output = self.dropout(output)
        output = self.layer_norm(output + residual)

        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None, clear_mask=None):
        # 1. dot product with weight matrices
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # 2. split tensor by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. do scale dot product to compute similarity
        out, attention = self.attention.forward(
            q, k, v, mask=mask, clear_mask=clear_mask
        )

        # 4. concat and pass to linear layer
        out = self.concat(out)
        out = self.w_concat(out)

        # 5. visualize attention map
        # TODO : we should implement visualization

        return out

    def split(self, tensor):
        """
        split tensor by number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        # it is similar with group convolution (split by number of heads)

        return tensor

    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor


class ScaleDotProductAttention(nn.Module):
    """
    compute scale dot product attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, clear_mask=None):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()

        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(2, 3)  # transpose
        score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product

        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        if clear_mask is not None:
            score = score.masked_fill(clear_mask.unsqueeze(0).unsqueeze(0), -1e9)

        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)

        # 4. multiply with Value
        v = score @ v

        return v, score


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        # '-1' means last dimension.

        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out
