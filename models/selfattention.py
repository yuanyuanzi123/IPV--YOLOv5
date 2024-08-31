import torch
import torch.nn as nn
import numpy as np


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size), "Embed size needs to be div by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys=10, query=10, mask=1):
        #def forward(self, values, keys, query, mask):
        N = query.shape[0]
        M = query.shape[1]
        values_array = np.shape(np.array(values.detach().numpy()))
        tensor_length = len(values_array)
        # print(values.shape)

        # split embedding into self.heads pieces
        if tensor_length == 3:
            value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
            # print("reshape之前：", values.shape)
            values = values.reshape(N, value_len, self.heads, self.head_dim)
            keys = keys.reshape(N, key_len, self.heads, self.head_dim)
            queries = query.reshape(N, query_len, self.heads, self.head_dim)
            # print("reshape之后：", values.shape)
        elif tensor_length == 4:
            value_len, key_len, query_len = values.shape[2], keys.shape[2], query.shape[2]
            # print("reshape之前：", values.shape)
            values = values.reshape(N, M, value_len, self.heads, self.head_dim)
            keys = keys.reshape(N, M, key_len, self.heads, self.head_dim)
            queries = query.reshape(N, M, query_len, self.heads, self.head_dim)
            # print("reshape之后：", values.shape)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        if tensor_length == 3:
            energy = torch.einsum("nqhd,nkhd->nhqk", queries, keys)
            # print("energy:", energy.shape)
            # queries shape: (N, query_len, heads, heads_dim)
            # keys shape : (N, key_len, heads, heads_dim)
            # energy shape: (N, heads, query_len, key_len)
        elif tensor_length == 4:
            energy = torch.einsum("nmqhd,nmkhd->nmhqk", queries, keys)
            # queries shape: (N, M, query_len, heads, heads_dim)
            # keys shape : (N, M, key_len, heads, heads_dim)
            # energy shape: (N, M, heads, query_len, key_len)


        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        if tensor_length == 3:
            out = torch.einsum("nhql, nlhd->nqhd", [attention, values]).reshape(N, query_len, self.heads * self.head_dim)
            # attention shape: (N, heads, query_len, key_len)
            # values shape: (N, value_len, heads, heads_dim)
            # (N, query_len, heads, head_dim)
        elif tensor_length == 4:
            out = torch.einsum("nmhql, nmlhd->nmqhd", [attention, values]).reshape(N, M, query_len, self.heads * self.head_dim)
            # attention shape: (N, M, heads, query_len, key_len)
            # values shape: (N, M, value_len, heads, heads_dim)
            # (N, M, query_len, heads, head_dim)

        out = self.fc_out(out)
        return out
