"""
从零实现一个 decoder-only Transformer（GPT 结构的最小版本）。
刻意不使用 nn.TransformerEncoderLayer, 把每一步矩阵运算展开, 便于观察学习流程。

一次前向的完整数据流（B=batch, T=序列长, V=27, D=d_model）:

    token ids            (B, T)          整数
      │  嵌入矩阵 E: (V, D)  —— 这就是"27 个 token 映射到的矩阵"
      ▼
    token emb            (B, T, D)
      │  + 位置嵌入 P: (T, D)   —— 告诉模型"第几个字符"
      ▼
    x                    (B, T, D)
      │  ┌─ LayerNorm -> 多头自注意力(带因果掩码) ─┐ 残差相加
      │  └─ LayerNorm -> 前馈网络 FFN            ─┘ 残差相加     × n_layer
      ▼
    x                    (B, T, D)
      │  LayerNorm -> 输出头 W_out: (D, V)
      ▼
    logits               (B, T, V)   每个位置对下一个字符的打分
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import VOCAB_SIZE


class MultiHeadSelfAttention(nn.Module):
    """多头因果自注意力: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k) + mask) V"""

    def __init__(self, d_model, n_head, block_size, dropout=0.1):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_head = d_model // n_head

        # 三个投影矩阵合并成一个 (D -> 3D), 一次矩阵乘法算出 Q K V
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model)          # 多头拼接后的输出投影
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        # 因果掩码: 下三角为 1, 保证第 t 个位置只能看到 <= t 的位置(不能偷看未来)
        mask = torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        self.register_buffer('mask', mask)

    def forward(self, x, return_attn=False):
        B, T, D = x.shape
        # (B,T,3D) -> 3 个 (B,T,D)
        q, k, v = self.qkv(x).split(D, dim=2)
        # 拆成多头: (B,T,D) -> (B, n_head, T, d_head)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)

        # 注意力分数 (B, n_head, T, T): 每个位置对其它位置的"关注度"
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)                      # 每行和为 1
        att_w = att                                       # 留给可视化
        att = self.attn_drop(att)

        y = att @ v                                       # (B, n_head, T, d_head) 加权求和
        y = y.transpose(1, 2).contiguous().view(B, T, D)   # 多头拼回 (B,T,D)
        y = self.resid_drop(self.proj(y))
        return (y, att_w) if return_attn else (y, None)


class FeedForward(nn.Module):
    """逐位置前馈网络: D -> 4D -> D, 负责在每个位置上做非线性特征变换"""

    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """一个 Transformer 层 = 注意力子层 + 前馈子层, 均为 Pre-LN + 残差"""

    def __init__(self, d_model, n_head, block_size, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, dropout)

    def forward(self, x, return_attn=False):
        a, att = self.attn(self.ln1(x), return_attn)
        x = x + a                      # 残差连接: 保证梯度能直通到底层
        x = x + self.ffn(self.ln2(x))
        return x, att


class MiniTransformer(nn.Module):
    def __init__(self, d_model=64, n_head=4, n_layer=2, block_size=32, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        # === 题目要求的"27 个 token 映射到的矩阵" ===
        self.tok_emb = nn.Embedding(VOCAB_SIZE, d_model)     # E: (27, D)
        self.pos_emb = nn.Embedding(block_size, d_model)     # P: (T, D) 可学习位置编码
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(d_model, n_head, block_size, dropout)
                                     for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, VOCAB_SIZE, bias=False)   # D -> 27
        # 权重绑定: 输出矩阵复用嵌入矩阵(经典技巧, 省参数且让"写"和"读"用同一套语义空间)
        self.head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, return_attn=False):
        B, T = idx.shape
        assert T <= self.block_size, f'序列长度 {T} 超过 block_size {self.block_size}'
        pos = torch.arange(T, device=idx.device)

        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))     # (B,T,D)
        attns = []
        for blk in self.blocks:
            x, att = blk(x, return_attn)
            if return_attn:
                attns.append(att)
        x = self.ln_f(x)
        logits = self.head(x)                                    # (B,T,27)

        loss = None
        if targets is not None:
            # 交叉熵: 把 (B,T,27) 摊平成 (B*T,27) 与 (B*T,) 对齐
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.reshape(-1))
        return logits, loss, attns

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """自回归采样: 每次用最后一个位置的分布采一个 token, 再拼回输入"""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]          # 只保留最近 block_size 个字符
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature       # 取最后一个位置 (B,27)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
        self.train()
        return idx

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


if __name__ == '__main__':
    m = MiniTransformer()
    x = torch.randint(0, VOCAB_SIZE, (2, 16))
    y = torch.randint(0, VOCAB_SIZE, (2, 16))     # 随机目标, 用于验证初始 loss ~= ln(27)
    logits, loss, attns = m(x, y, return_attn=True)
    print(f'参数量        : {m.n_params():,}')
    print(f'嵌入矩阵 E    : {tuple(m.tok_emb.weight.shape)}   <- 27 个 token 各占一行')
    print(f'logits        : {tuple(logits.shape)}')
    print(f'初始 loss     : {loss.item():.4f}  (随机猜测的理论值 ln(27) = {math.log(27):.4f})')
    print(f'注意力图      : {len(attns)} 层, 每层 {tuple(attns[0].shape)} = (B, head, T, T)')
