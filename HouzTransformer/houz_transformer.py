#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: winku
@Date: 2025/3/19 18:10
@Description: 构建Transformer
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F


'''注意力机制计算函数'''
def attention(q, k, v, dropout_module = None, is_causal=False, dropout=None, mask=None):
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    if is_causal:
        att = att.masked_fill(mask[:, :, :k.size(-2), :k.size(-2)] == 0, float('-inf'))
    att = F.softmax(att, dim=-1)
    att = dropout_module(att)
    y = att @ v
    return y


'''多头注意力机制计算模块'''
class MultiHeadAttention(nn.Module):

    def __init__(self, config, is_causal=False):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attns = nn.ModuleList([nn.Linear(config.n_embd, config.n_embd, bias=config.bias) for _ in range(3)])
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        # 头数
        self.n_head = config.n_head
        # 隐藏层维度
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.is_causal = is_causal
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

        if not self.flash:
            print("WARNING: using slow attention. Flash Attention require PyTorch >= 2.0")
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                 .view(1, 1, config.block_size, config.block_size))

    def forward(self, query, key, value):
        B, T, C = query.size()
        q, k, v = [self.c_attns[i](x) for i, x, in zip(range(3), (query, key, value))]
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # 注意力计算
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=self.is_causal)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if self.is_causal:
                att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y
