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
from functools import lru_cache
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from nemo.collections.common.parts.training_utils import avoid_float16_autocast_context

__all__ = [
    'ChunkedRelPositionalEncoding',
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

    def __init__(self, n_head, n_feat, dropout_rate, max_cache_len=0):
        """Construct an MultiHeadedAttention object."""
        super(MultiHeadAttention, self).__init__()
        self.cache_drop_size = None
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

        self._max_cache_len = max_cache_len
        self._cache_id = None

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

    def forward(self, query, key, value, mask, pos_emb=None, cache=None, cache_next=None):
        """Compute 'Scaled Dot Product Attention'.
        Args:
            query (torch.Tensor): (batch, time1, size)
            key (torch.Tensor): (batch, time2, size)
            value(torch.Tensor): (batch, time2, size)
            mask (torch.Tensor): (batch, time1, time2)
            cache (torch.Tensor) : (cache_nums, batch, time_cache, size)
            cache_next (torch.Tensor) : (cache_nums, batch, time_cache_next, size)

        returns:
            output (torch.Tensor): transformed `value` (batch, time1, d_model) weighted by the query dot key attention
        """
        key, value, query = self.update_cache(key=key, value=value, query=query, cache=cache, cache_next=cache_next)

        # temporary until we solve this more gracefully
        with avoid_float16_autocast_context():
            q, k, v = self.forward_qkv(query, key, value)
            scores = torch.matmul(q, k.transpose(-2, -1)) / self.s_d_k
            out = self.forward_attention(v, scores, mask)

        return out

    def update_cache(self, key, value, query, cache, cache_next):
        if cache is not None:
            q_length = query.size(1)
            q_input = query
            key = value = torch.cat((cache[self._cache_id], key), dim=1)

        if cache_next is not None:
            cache_next_length = cache_next.size(2)
            q_keep_size = q_length - self.cache_drop_size

            cache_next[self._cache_id, :, :-q_keep_size, :] = cache[
                self._cache_id, :, -(cache_next_length - q_keep_size) :, :
            ].clone()
            cache_next[self._cache_id, :, -q_keep_size:, :] = q_input[:, :q_keep_size, :]

        return key, value, query


class RelPositionMultiHeadAttention(MultiHeadAttention):
    """Multi-Head Attention layer of Transformer-XL with support of relative positional encoding.
    Paper: https://arxiv.org/abs/1901.02860
    Args:
        n_head (int): number of heads
        n_feat (int): size of the features
        dropout_rate (float): dropout rate
    """

    def __init__(self, n_head, n_feat, dropout_rate, pos_bias_u, pos_bias_v, max_cache_len=0, att_type='rel_pos'):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(n_head=n_head, n_feat=n_feat, dropout_rate=dropout_rate, max_cache_len=max_cache_len)
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

    def forward(self, query, key, value, mask, pos_emb, cache=None, cache_next=None):
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.
        Args:
            query (torch.Tensor): (batch, time1, size)
            key (torch.Tensor): (batch, time2, size)
            value(torch.Tensor): (batch, time2, size)
            mask (torch.Tensor): (batch, time1, time2)
            pos_emb (torch.Tensor) : (batch, time1, size)
            cache (torch.Tensor) : (cache_nums, batch, time_cache, size)
            cache_next (torch.Tensor) : (cache_nums, batch, time_cache_next, size)
        Returns:
            output (torch.Tensor): transformed `value` (batch, time1, d_model) weighted by the query dot key attention
        """
        key, value, query = self.update_cache(key=key, value=value, query=query, cache=cache, cache_next=cache_next)

        # temporary until we solve this more gracefully
        with avoid_float16_autocast_context():
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
            out = self.forward_attention(v, scores, mask)

        return out

    def _chunk_nooverlap(self, x, left_chunk_size, chunk_size, right_chunk_size, pad_value=0.):
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
        stride = list(x.stride())
        output_stride = stride[:2] + [chunk_size * D] + stride[-2:]
        return x.as_strided(size=output_size, stride=output_stride)

    def rel_shift_nooverlap(self, x):
        """Compute relative positional encoding.
        Args:
            x (torch.Tensor): (batch, nheads, time, 2*time-1)
        """
        b, h, t, qlen, pos_len = x.size()  # (b, h, t1, t2)
        # need to add a column of zeros on the left side of last dimension to perform the relative shifting
        x = torch.nn.functional.pad(x, pad=(1, 0))  # (b, h, t1, t2+1)
        x = x.view(b, h, t, -1, qlen)  # (b, h, t2+1, t1)
        # need to drop the first row
        x = x[:, :, :, 1:].view(b, h, t, qlen, pos_len)  # (b, h, t1, t2)
        return x

    def forward_attention_nooverlap(self, value, scores, mask):
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


    def forward_nooverlap(self, query, key, value, mask, pos_emb, left_chunk_size=None, chunk_size=None, right_chunk_size=None):
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
        assert query.size(1) == key.size(1)
        q, k, v = self.forward_qkv(query, key, value)

        # save original length
        n_batch, _, T, _ = q.size()
        padding_len = chunk_size - T % chunk_size if T % chunk_size else 0
        q = F.pad(q, (0, 0, 0, padding_len))
        k = F.pad(k, (0, 0, 0, padding_len))
        v = F.pad(v, (0, 0, 0, padding_len))

        # B x H x T // chunk_size x total_chunk_size x D
        k = self._chunk_nooverlap(k, left_chunk_size, chunk_size, right_chunk_size)
        v = self._chunk_nooverlap(v, left_chunk_size, chunk_size, right_chunk_size)

        q = q.reshape(n_batch, self.h, (T + chunk_size - 1) // chunk_size, chunk_size, self.d_k)
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
        matrix_bd = self.rel_shift_nooverlap(matrix_bd)
        # drops extra elements in the matrix_bd to match the matrix_ac's size
        matrix_bd = matrix_bd[:, :, :, :, : matrix_ac.size(-1)]

        # B x H x T // chunk_size x chunk_size x total_chunk_size
        scores = (matrix_ac + matrix_bd) / self.s_d_k

        # expand shape to _chunk
        # B x 1 x T x 1
        mask = F.pad(mask, (0, padding_len), value=1.).view(n_batch, 1, -1, 1)
        mask = self._chunk_nooverlap(mask, left_chunk_size, chunk_size, right_chunk_size, pad_value=1.)
        # B x T // chunk size x 1 x total_chunk size
        mask = mask.view(n_batch, mask.size(2), 1, mask.size(3))
        return self.forward_attention_nooverlap(v, scores, mask)[:,:T]

    # Longformer implementation for no overlap case adapted for arbitrary left and right chunk size
    # https://github.com/allenai/longformer/blob/master/longformer/sliding_chunks.py
    def _skew(self, x, direction, padding_value):
        '''Convert diagonals into columns (or columns into diagonals depending on `direction`'''
        x_padded = F.pad(x, direction, value=padding_value)
        x_padded = x_padded.view(*x_padded.size()[:-2], x_padded.size(-1), x_padded.size(-2))
        return x_padded
    
    
    def _skew2(self, x, padding_value):
        '''shift every row 1 step to right converting columns into diagonals'''
        # X = B x C x M x L
        B, C, M, L = x.size()
        x = F.pad(x, (0, M + 1), value=padding_value)  # B x C x M x (L+M+1)
        x = x.view(B, C, -1)  # B x C x ML+MM+M
        x = x[:, :, :-M]  # B x C x ML+MM
        x = x.view(B, C, M, M + L)  # B x C, M x L+M
        x = x[:, :, :, :-1]
        return x
    
    
    def _chunk_overlap(self, x, w):
        '''convert into overlapping chunkings. Chunk size = 2w, overlap size = w'''
    
        # non-overlapping chunks of size = 2w
        x = x.view(x.size(0), x.size(1) // (w * 2), w * 2, x.size(2))
    
        # use `as_strided` to make the chunks overlap with an overlap size = w
        chunk_size = list(x.size())
        chunk_size[1] = chunk_size[1] * 2 - 1
    
        chunk_stride = list(x.stride())
        chunk_stride[1] = chunk_stride[1] // 2
        return x.as_strided(size=chunk_size, stride=chunk_stride)


    def _get_invalid_locations_mask_fixed_dilation(self, seq_len: int, w: int, d: int):
        diagonals_list = []
        for j in range(-d * w, d, d):
            diagonal_mask = torch.zeros(seq_len, device='cpu', dtype=torch.uint8)
            diagonal_mask[:-j] = 1
            diagonals_list.append(diagonal_mask)
        return torch.stack(diagonals_list, dim=-1)
    
    @lru_cache()
    def _get_invalid_locations_mask(self, w: int, d: Union[torch.Tensor,int], autoregressive: bool, device: str):
        if isinstance(d, int):
            affected_seq_len = w * d
            mask = self._get_invalid_locations_mask_fixed_dilation(affected_seq_len, w, d)
            mask = mask[None, None, :, :]
        else:
            # TODO not verified
            affected_seq_len = w * d.max()
            head_masks = []
            d_list = d.cpu().numpy().tolist()
            for d in d_list:
                one_head_mask = _get_invalid_locations_mask_fixed_dilation(affected_seq_len, w, d)
                head_masks.append(one_head_mask)
            mask = torch.stack(head_masks, dim=-2)
            mask = mask[None, :, :, :]
    
        ending_mask = None if autoregressive else mask.flip(dims=(2, 3)).bool().to(device)
        return affected_seq_len, mask.bool().to(device), ending_mask
    
    def mask_invalid_locations(self, input_tensor: torch.Tensor, w: int, d: Union[torch.Tensor, int], autoregressive: bool) -> torch.Tensor:
        affected_seq_len, beginning_mask, ending_mask = self._get_invalid_locations_mask(w, d, autoregressive, input_tensor.device)
        seq_len = input_tensor.size(2)
        # NOTE shape diff from original code
        beginning_input = input_tensor[:, :, :affected_seq_len, :w+1]
        beginning_mask = beginning_mask[:, :, :seq_len].expand(beginning_input.size())
        beginning_input.masked_fill_(beginning_mask, -float('inf'))
        if not autoregressive:
            ending_input = input_tensor[:, :, -affected_seq_len:, -(w+1):]
            ending_mask = ending_mask[:, :, -seq_len:].expand(ending_input.size())
            ending_input.masked_fill_(ending_mask, -float('inf'))


    def sliding_chunks_matmul_qk(self, q: torch.Tensor, k: torch.Tensor, w: int, padding_value: float):
        '''Matrix multiplicatio of query x key tensors using with a sliding window attention pattern.
        This implementation splits the input into overlapping chunks of size 2w (e.g. 512 for pretrained Longformer)
        with an overlap of size w'''
        bsz, num_heads, seqlen, head_dim = q.size()
        assert seqlen % (w * 2) == 0
        assert q.size() == k.size()
    
        chunks_count = seqlen // w - 1
    
        # group bsz and num_heads dimensions into one, then chunk seqlen into chunks of size w * 2
        q = q.reshape(bsz * num_heads, seqlen, head_dim)
        k = k.reshape(bsz * num_heads, seqlen, head_dim)
    
        chunk_q = self._chunk_overlap(q, w)
        chunk_k = self._chunk_overlap(k, w)
    
        # matrix multipication
        # bcxd: bsz*num_heads x chunks x 2w x head_dim
        # bcyd: bsz*num_heads x chunks x 2w x head_dim
        # bcxy: bsz*num_heads x chunks x 2w x 2w
        chunk_attn = torch.einsum('bcxd,bcyd->bcxy', (chunk_q, chunk_k))  # multiply
    
        # convert diagonals into columns
        diagonal_chunk_attn = self._skew(chunk_attn, direction=(0, 0, 0, 1), padding_value=padding_value)
    
        # allocate space for the overall attention matrix where the chunks are compined. The last dimension
        # has (w * 2 + 1) columns. The first (w) columns are the w lower triangles (attention from a word to
        # w previous words). The following column is attention score from each word to itself, then
        # followed by w columns for the upper triangle.
    
        diagonal_attn = diagonal_chunk_attn.new_empty((bsz * num_heads, chunks_count + 1, w, w * 2 + 1))
    
        # copy parts from diagonal_chunk_attn into the compined matrix of attentions
        # - copying the main diagonal and the upper triangle
        diagonal_attn[:, :-1, :, w:] = diagonal_chunk_attn[:, :, :w, :w + 1]
        diagonal_attn[:, -1, :, w:] = diagonal_chunk_attn[:, -1, w:, :w + 1]
        # - copying the lower triangle
        diagonal_attn[:, 1:, :, :w] = diagonal_chunk_attn[:, :, - (w + 1):-1, w + 1:]
        diagonal_attn[:, 0, 1:w, 1:w] = diagonal_chunk_attn[:, 0, :w - 1, 1 - w:]
    
        # separate bsz and num_heads dimensions again
        diagonal_attn = diagonal_attn.view(bsz, num_heads, seqlen, 2 * w + 1)
    
        self.mask_invalid_locations(diagonal_attn, w, 1, False)
        return diagonal_attn

    def sliding_chunks_matmul_pv(self, prob: torch.Tensor, v: torch.Tensor, w: int):
        '''Same as sliding_chunks_matmul_qk but for prob and value tensors. It is expecting the same output
        format from sliding_chunks_matmul_qk'''
        bsz, num_heads, seqlen, head_dim = v.size()
        assert seqlen % (w * 2) == 0
        assert prob.size()[:3] == v.size()[:3]
        assert prob.size(3) == 2 * w + 1
        chunks_count = seqlen // w - 1
        # group bsz and num_heads dimensions into one, then chunk seqlen into chunks of size 2w
        chunk_prob = prob.reshape(bsz * num_heads, seqlen // w, w, 2 * w + 1)
    
        # group bsz and num_heads dimensions into one
        v = v.reshape(bsz * num_heads, seqlen, head_dim)
    
        # pad seqlen with w at the beginning of the sequence and another w at the end
        padded_v = F.pad(v, (0, 0, w, w), value=-1)
    
        # chunk padded_v into chunks of size 3w and an overlap of size w
        chunk_v_size = (bsz * num_heads, chunks_count + 1, 3 * w, head_dim)
        chunk_v_stride = padded_v.stride()
        chunk_v_stride = chunk_v_stride[0], w * chunk_v_stride[1], chunk_v_stride[1], chunk_v_stride[2]
        chunk_v = padded_v.as_strided(size=chunk_v_size, stride=chunk_v_stride)
    
        skewed_prob = self._skew2(chunk_prob, padding_value=0)
    
        context = torch.einsum('bcwd,bcdh->bcwh', (skewed_prob, chunk_v))
        return context.view(bsz, num_heads, seqlen, head_dim).transpose(1, 2)

    def forward_overlap(self, query, key, value, mask, pos_emb, left_context_size=None, right_context_size=None):
        q, k, v = self.forward_qkv(query, key, value)
        n_batch, _, T, _ = q.size()

        w = max(left_context_size, right_context_size)
        assert w > 0
        pad_len = (2 * w - T % (2 * w)) % (2 * w)
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
        mask = F.pad(mask, (0, pad_len), value=1.)

        # B x H x T x D
        q_with_bias_u = (q + self.pos_bias_u.unsqueeze(1))
        q_with_bias_v = (q + self.pos_bias_v.unsqueeze(1))

        diagonal_matrix_ac = self.sliding_chunks_matmul_qk(q_with_bias_u, k, w, padding_value=0.)
        # add relative positional embedding

        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k).transpose(1, 2)
        # B x H x pos_emb_len x D
        diagonal_matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))

        start_pos = w - left_context_size
        end_pos = w + right_context_size

        # B x H x T // chunk_size x chunk_size x total_chunk_size
        diagonal_matrix_ac[:, :, :, :left_context_size] += diagonal_matrix_bd[:, :, :, :left_context_size]
        diagonal_matrix_ac[:, :, :, -(right_context_size + 1):] += diagonal_matrix_bd[:, :, :, left_context_size:]
        scores = diagonal_matrix_ac / self.s_d_k

        # mask invalid positions
        scores[:, :, :, :start_pos] = -10000.0
        scores[:, :, :, end_pos+1:] = -10000.0

        # This implementation is fast and takes very little memory because num_heads x hidden_size = 1
        # from (bsz x seq_len) to (bsz x num_heads x seqlen x hidden_size)
        mask = mask.unsqueeze(dim=1).unsqueeze(dim=-1)
        # cast to float/half then replace 1's with -inf
        float_mask = mask.type_as(scores).masked_fill(mask, -10000.0)
        ones = float_mask.new_ones(size=float_mask.size())  # tensor of ones
        # diagonal mask with zeros everywhere and -inf inplace of padding
	# TODO what pad value to use?
        d_mask = self.sliding_chunks_matmul_qk(ones, float_mask, w, padding_value=0.)

        scores += d_mask

        attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)
        p_attn = self.dropout(attn)

        x = self.sliding_chunks_matmul_pv(p_attn, v, w).reshape(n_batch, -1, self.h * self.d_k)[:, :T]
        return self.linear_out(x)

    def forward(self, query, key, value, mask, pos_emb, left_chunk_size=None, chunk_size=None, right_chunk_size=None):
        if self.att_type == 'longformer_chunked_rel_pos':
            return self.forward_nooverlap(query, key, value, mask, pos_emb, left_chunk_size=left_chunk_size, chunk_size=chunk_size, right_chunk_size=right_chunk_size)
        elif self.att_type == 'longformer_overlap_rel_pos':
            return self.forward_overlap(query, key, value, mask, pos_emb, left_context_size=left_chunk_size, right_context_size=right_chunk_size)
        else:
            assert self.att_type == 'rel_pos'
            return self.forward_rel_pos(query, key, value, mask, pos_emb)


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

    def forward(self, x, cache_len=0):
        """Compute positional encoding.
        Args:
            x (torch.Tensor): Input. Its shape is (batch, time, feature_size)
            cache_len (int): the size of the cache which is used to shift positions
        Returns:
            x (torch.Tensor): Its shape is (batch, time, feature_size)
            pos_emb (torch.Tensor): Its shape is (1, time, feature_size)
        """

        if self.xscale:
            x = x * self.xscale

        # center_pos would be the index of position 0
        # negative positions would be used for right and positive for left tokens
        # for input of length L, 2*L-1 positions are needed, positions from (L-1) to -(L-1)
        input_len = x.size(1) + cache_len
        start_pos = self.center_pos - input_len
        end_pos = self.center_pos + input_len - 1
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
        super(ChunkedRelPositionalEncoding, self).__init__(**kwargs)
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
