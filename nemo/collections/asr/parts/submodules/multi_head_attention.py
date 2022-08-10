# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
Part of this code is adopted from https://github.com/espnet/espnet
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'RelPositionMultiHeadAttention',
    'RelPositionalEncoding',
    'PositionalEncoding',
]


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention layer of Transformer.
    Args:
        n_head (int): number of heads
        n_feat (int): size of the features
        dropout_rate (float): dropout rate
    """

    def __init__(self, n_head, n_feat, dropout_rate):
        """Construct an MultiHeadedAttention object."""
        super(MultiHeadAttention, self).__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.s_d_k = math.sqrt(self.d_k)
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward_qkv(self, query, key, value):
        """Transforms query, key and value.
        Args:
            query (torch.Tensor): (batch, time1, size)
            key (torch.Tensor): (batch, time2, size)
            value (torch.Tensor): (batch, time2, size)
        returns:
            q (torch.Tensor): (batch, head, time1, size)
            k (torch.Tensor): (batch, head, time2, size)
            v (torch.Tensor): (batch, head, time2, size)
        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        return q, k, v

    def forward_attention(self, value, scores, mask):
        """Compute attention context vector.
        Args:
            value (torch.Tensor): (batch, time2, size)
            scores(torch.Tensor): (batch, time1, time2)
            mask(torch.Tensor): (batch, time1, time2)
        returns:
            value (torch.Tensor): transformed `value` (batch, time2, d_model) weighted by the attention scores
        """
        n_batch = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1)  # (batch, 1, time1, time2)
            scores = scores.masked_fill(mask, -10000.0)
            attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)  # (batch, head, time1, time2)
        else:
            attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = x.transpose(1, 2).reshape(n_batch, -1, self.h * self.d_k)  # (batch, time1, d_model)

        return self.linear_out(x)  # (batch, time1, d_model)

    def forward(self, query, key, value, mask, pos_emb=None):
        """Compute 'Scaled Dot Product Attention'.
        Args:
            query (torch.Tensor): (batch, time1, size)
            key (torch.Tensor): (batch, time2, size)
            value(torch.Tensor): (batch, time2, size)
            mask (torch.Tensor): (batch, time1, time2)
        returns:
            output (torch.Tensor): transformed `value` (batch, time1, d_model) weighted by the query dot key attention
        """
        q, k, v = self.forward_qkv(query, key, value)
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.s_d_k
        return self.forward_attention(v, scores, mask)


class RelPositionMultiHeadAttention(MultiHeadAttention):
    """Multi-Head Attention layer of Transformer-XL with support of relative positional encoding.
    Paper: https://arxiv.org/abs/1901.02860
    Args:
        n_head (int): number of heads
        n_feat (int): size of the features
        dropout_rate (float): dropout rate
    """

    def __init__(self, n_head, n_feat, dropout_rate, pos_bias_u, pos_bias_v):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(n_head, n_feat, dropout_rate)
        # linear transformation for positional encoding
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        # these two learnable biases are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        if pos_bias_u is None or pos_bias_v is None:
            self.pos_bias_u = nn.Parameter(torch.FloatTensor(self.h, self.d_k))
            self.pos_bias_v = nn.Parameter(torch.FloatTensor(self.h, self.d_k))
            # nn.init.normal_(self.pos_bias_u, 0.0, 0.02)
            # nn.init.normal_(self.pos_bias_v, 0.0, 0.02)
            nn.init.zeros_(self.pos_bias_u)
            nn.init.zeros_(self.pos_bias_v)
        else:
            self.pos_bias_u = pos_bias_u
            self.pos_bias_v = pos_bias_v

    def rel_shift(self, x):
        """Compute relative positional encoding.
        Args:
            x (torch.Tensor): (batch, nheads, time, 2*time-1)
        """
        b, h, qlen, pos_len = x.size()  # (b, h, t1, t2)
        # need to add a column of zeros on the left side of last dimension to perform the relative shifting
        x = torch.nn.functional.pad(x, pad=(1, 0))  # (b, h, t1, t2+1)
        x = x.view(b, h, -1, qlen)  # (b, h, t2+1, t1)
        # need to drop the first row
        x = x[:, :, 1:].view(b, h, qlen, pos_len)  # (b, h, t1, t2)
        return x

    def forward(self, query, key, value, mask, pos_emb):
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.
        Args:
            query (torch.Tensor): (batch, time1, size)
            key (torch.Tensor): (batch, time2, size)
            value(torch.Tensor): (batch, time2, size)
            mask (torch.Tensor): (batch, time1, time2)
            pos_emb (torch.Tensor) : (batch, time1, size)
        Returns:
            output (torch.Tensor): transformed `value` (batch, time1, d_model) weighted by the query dot key attention
        """
        q, k, v = self.forward_qkv(query, key, value)
        q = q.transpose(1, 2)  # (batch, time1, head, d_k)

        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1, 2)  # (batch, head, time1, d_k)

        # (batch, head, time1, d_k)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        # (batch, head, time1, d_k)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch, head, time1, time2)
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

        # compute matrix b and matrix d
        # (batch, head, time1, time2)
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        matrix_bd = self.rel_shift(matrix_bd)
        # drops extra elements in the matrix_bd to match the matrix_ac's size
        matrix_bd = matrix_bd[:, :, :, : matrix_ac.size(-1)]

        scores = (matrix_ac + matrix_bd) / self.s_d_k  # (batch, head, time1, time2)

        return self.forward_attention(v, scores, mask)


class PositionalEncoding(torch.nn.Module):
    """Fixed sinusoidal positional encoding.
    Args:
        d_model (int): embedding dim
        dropout_rate (float): dropout rate
        max_len (int): maximum input length
        xscale (bool): whether to scale the input by sqrt(d_model)
        dropout_rate_emb (float): dropout rate for the positional embeddings
    """

    def __init__(self, d_model, dropout_rate, max_len=5000, xscale=None, dropout_rate_emb=0.0):
        """Construct an PositionalEncoding object."""
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.xscale = xscale
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.max_len = max_len
        if dropout_rate_emb > 0:
            self.dropout_emb = nn.Dropout(dropout_rate_emb)
        else:
            self.dropout_emb = None

    def create_pe(self, positions):
        pos_length = positions.size(0)
        pe = torch.zeros(pos_length, self.d_model, device=positions.device)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32, device=positions.device)
            * -(math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        pe = pe.unsqueeze(0)
        if hasattr(self, 'pe'):
            self.pe = pe
        else:
            self.register_buffer('pe', pe, persistent=False)

    def extend_pe(self, length, device):
        """Reset and extend the positional encodings if needed."""
        if hasattr(self, 'pe') and self.pe.size(1) >= length:
            return
        positions = torch.arange(0, length, dtype=torch.float32, device=device).unsqueeze(1)
        self.create_pe(positions=positions)

    def forward(self, x: torch.Tensor):
        """Adds positional encoding.
        Args:
            x (torch.Tensor): Input. Its shape is (batch, time, feature_size)
        Returns:
            x+pos_emb (torch.Tensor): Its shape is (batch, time, feature_size)
            pos_emb (torch.Tensor): Its shape is (1, time, feature_size)
        """
        if self.xscale:
            x = x * self.xscale
        pos_emb = self.pe[:, : x.size(1)]
        if self.dropout_emb:
            pos_emb = self.dropout_emb(pos_emb)
        x = x + pos_emb
        return self.dropout(x), pos_emb


class RelPositionalEncoding(PositionalEncoding):
    """Relative positional encoding for TransformerXL's layers
    See : Appendix B in https://arxiv.org/abs/1901.02860
    Args:
        d_model (int): embedding dim
        dropout_rate (float): dropout rate
        max_len (int): maximum input length
        xscale (bool): whether to scale the input by sqrt(d_model)
        dropout_rate_emb (float): dropout rate for the positional embeddings
    """

    def extend_pe(self, length, device):
        """Reset and extend the positional encodings if needed."""
        needed_size = 2 * length - 1
        if hasattr(self, 'pe') and self.pe.size(1) >= needed_size:
            return
        # positions would be from negative numbers to positive
        # positive positions would be used for left positions and negative for right positions
        positions = torch.arange(length - 1, -length, -1, dtype=torch.float32, device=device).unsqueeze(1)
        self.create_pe(positions=positions)
        self.center_pos = torch.tensor(self.pe.size(1) // 2 + 1, dtype=torch.int32, device=device)

    def forward(self, x):
        """Compute positional encoding.
        Args:
            x (torch.Tensor): Input. Its shape is (batch, time, feature_size)
        Returns:
            x (torch.Tensor): Its shape is (batch, time, feature_size)
            pos_emb (torch.Tensor): Its shape is (1, time, feature_size)
        """

        if self.xscale:
            x = x * self.xscale

        # center_pos would be the index of position 0
        # negative positions would be used for right and positive for left tokens
        # for input of length L, 2*L-1 positions are needed, positions from (L-1) to -(L-1)
        start_pos = self.center_pos - x.size(1)
        end_pos = self.center_pos + x.size(1) - 1
        pos_emb = self.pe[:, start_pos:end_pos]
        if self.dropout_emb:
            pos_emb = self.dropout_emb(pos_emb)
        return self.dropout(x), pos_emb

class ChunkedRelPositionalEncoding(PositionalEncoding):
    """Relative positional encoding for Longformer's non-overlapping
    sliding window attention or chunked attention. See above for relative
    positional encoding based on Transformer-XL paper
    Args:
        left_chunk_size (int): number of frames to in past chunks
        chunk size (int): number of frames (max frames if using multimode) in current chunk
        d_model (int): embedding dim
        dropout_rate (float): dropout rate
        max_len (int): maximum input length
        xscale (bool): whether to scale the input by sqrt(d_model)
        dropout_rate_emb (float): dropout rate for the positional embeddings
    """
    def __init__(self, left_chunk_size, chunk_size, right_chunk_size, **kwargs):
        super(self, ChunkedRelPositionalEncoding).__init__(**kwargs)
        self.left_chunk_size = left_chunk_size
        self.chunk_size = chunk_size
        self.right_chunk_size = right_chunk_size

    def extend_pe(self, length, device):
        """Reset and extend the positional encodings only at the beginning
        as chunk size and left chunk size does not change."""
        if hasattr(self, 'pe'):
            return

        # ignore passed length
        max_left_context_size = self.left_chunk_size + self.chunk_size - 1
        max_right_context_size = self.chunk_size + self.right_chunk_size - 1
        positions = torch.arange(max_left_context_size, -max_right_context_size - 1, -1, dtype=torch.float32, device=device).unsqueeze(1)
        self.create_pe(positions=positions)
        self.center_pos = torch.tensor(max_left_context_size, dtype=torch.int32, device=device)

    def forward(self, x, left_chunk_size=None, chunk_size=None, right_chunk_size=None):
        """Compute positional encoding.
        Args:
            x (torch.Tensor): Input. Its shape is (batch, time, feature_size)
        Returns:
            x (torch.Tensor): Its shape is (batch, time, feature_size)
            pos_emb (torch.Tensor): Its shape is (1, time, feature_size)
        """

        if self.xscale:
            x = x * self.xscale

        # center_pos would be the index of position 0
        # negative positions would be used for right and positive for left tokens
        # for input of length L, 2*L-1 positions are needed, positions from (L-1) to -(L-1)
        if left_chunk_size is None:
            left_chunk_size = self.left_chunk_size
        if chunk_size is None:
            chunk_size = self.chunk_size
        if right_chunk_size is None:
            right_chunk_size = self.right_chunk_size

        left_chunk_size += chunk_size - 1
        right_chunk_size += chunk_size - 1

        start_pos = self.center_pos - left_chunk_size
        end_pos = self.center_pos + right_chunk_size + 1
        pos_emb = self.pe[:, start_pos:end_pos]
        if self.dropout_emb:
            pos_emb = self.dropout_emb(pos_emb)
        return self.dropout(x), pos_emb

class ChunkedRelPositionMultiHeadAttention(RelPositionMultiHeadAttention):
    """Multi-Head Attention layer of Transformer-XL with support of relative positional encoding.
    Paper: https://arxiv.org/abs/1901.02860
    Args:
        n_head (int): number of heads
        n_feat (int): size of the features
        dropout_rate (float): dropout rate
    """
    def _chunk(self, x, left_chunk_size, chunk_size, right_chunk_size, pad_value=0.):
        """
        x: B x H x T x D
        output: B x H x T // chunk_size x total_chunk_size x D
        """
        B, H, T, D = x.size()
        assert T % chunk_size == 0
        total_chunk_size = left_chunk_size + chunk_size + right_chunk_size
        x = F.pad(x, (0, 0, left_chunk_size, right_chunk_size), value=pad_value) # B x H x T x D

        # unfold can be slow
        # x = x.unfold(dim=dim, size=total_chunk_size, step=chunk_size).transpose(-1, -2)
        # return x

        output_size = [B, H, T // chunk_size, total_chunk_size, D]
        output_stride = [
            H * (T // chunk_size) * total_chunk_size * D, (T // chunk_size) * total_chunk_size, chunk_size * D, D, 1
        ]
        return x.as_strided(size=output_size, stride=output_stride)

    def forward_attention(self, value, scores, mask):
        """Compute attention context vector.
        Args:
            value (torch.Tensor): (batch, time2, size)
            scores(torch.Tensor): (batch, time1, time2)
            mask(torch.Tensor): (batch, time1, time2)
        returns:
            value (torch.Tensor): transformed `value` (batch, time2, d_model) weighted by the attention scores
        """
        n_batch = value.size(0)
        if mask is not None:
             # B x 1 x T // chunk_size x 1 x total_chunk_size
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask, -10000.0)
            attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)
        else:
            attn = torch.softmax(scores, dim=-1)

        # (B, H, T // chunk size, chunk size, total_chunk_size)
        p_attn = self.dropout(attn)
        # (B, H, T // chunk size, chunk size, D)
        x = torch.matmul(p_attn, value)
        # B x T x D
        x = x.reshape(n_batch, self.h, -1, self.d_k).transpose(1, 2).reshape(n_batch, -1, self.h * self.d_k)

        return self.linear_out(x)

    def forward(self, query, key, value, mask, pos_emb, left_chunk_size=None, chunk_size=None, right_chunk_size=None):
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.
        Args:
            query (torch.Tensor): (batch, time1, size)
            key (torch.Tensor): (batch, time2, size)
            value(torch.Tensor): (batch, time2, size)
            mask (torch.Tensor): (batch, time1, time2)
            pos_emb (torch.Tensor) : (batch, time1, size)
        Returns:
            output (torch.Tensor): transformed `value` (batch, time1, d_model) weighted by the query dot key attention
        """
        # only pad mask is required
        assert len(mask.shape) == 2
        assert q.size(1) == k.size(1)
        q, k, v = self.forward_qkv(query, key, value)

        # save original length
        n_batch, _, T, _ = q.size()
        padding_len = chunk_size - T % chunk_size if T % chunk_size else 0
        q = F.pad(q, (0, 0, 0, padding_len))
        k = F.pad(k, (0, 0, 0, padding_len))
        v = F.pad(v, (0, 0, 0, padding_len))

        # B x H x T // chunk_size x total_chunk_size x D
        k = self._chunk(k, left_chunk_size, chunk_size, right_chunk_size)
        v = self._chunk(v, left_chunk_size, chunk_size, right_chunk_size)

        q = q.reshpae(n_batch, self.h, (T + chunk_size - 1) // chunk_size, chunk_size, self.d_k)
        q = q.transpose(1, 3)

        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        # B x H x 1 x pos_emb_len x D
        p = p.transpose(1, 2).unsqueeze(2)  

        # B x H x T // chunk_size x chunk_size x D
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 3)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 3)

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch, head, time1, time2)
        # B x H x T // chunk_size x chunk_size x total_chunk_size
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

        # compute matrix b and matrix d
        # B x H x T // chunk_size x chunk_size x pos_emb_len
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        matrix_bd = self.rel_shift(matrix_bd)
        # drops extra elements in the matrix_bd to match the matrix_ac's size
        matrix_bd = matrix_bd[:, :, :, : matrix_ac.size(-1)]

        # B x H x T // chunk_size x chunk_size x total_chunk_size
        scores = (matrix_ac + matrix_bd) / self.s_d_k

        # expand shape to _chunk
        # B x 1 x T x 1
        mask = F.pad(mask, (0, padding_len), value=1.).view(n_batch, 1, -1, 1)
        mask = self._chunk(mask, left_chunk_size, chunk_size, right_chunk_size, pad_value=1.)
        # B x T // chunk size x 1 x total_chunk size
        mask = mask.view(n_batch, mask.size(2), 1, mask.size(3))
        return self.forward_attention(v, scores, mask)[:,:T]
