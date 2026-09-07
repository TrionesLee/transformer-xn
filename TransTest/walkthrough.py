"""
逐步走查: 把一次前向传播 + 一次参数更新的每个中间结果都打印出来,
用来"看清" Transformer 内部到底发生了什么。

    python3 walkthrough.py                 # 用随机初始化的模型
    python3 walkthrough.py --ckpt ckpt.pt  # 用训练好的模型(能看到有意义的注意力)
"""

import argparse
import math
import torch
import torch.nn.functional as F

from data import encode, decode, one_hot, ITOS, VOCAB_SIZE
from model import MiniTransformer


def sep(title):
    print('\n' + '=' * 76)
    print(title)
    print('=' * 76)


def brief(t, n=6):
    flat = t.detach().flatten()[:n]
    return '[' + ', '.join(f'{x:+.3f}' for x in flat) + (', ...]' if t.numel() > n else ']')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--text', default='the cat')
    ap.add_argument('--ckpt', default=None)
    args = ap.parse_args()

    torch.manual_seed(0)
    if args.ckpt:
        from inspect_model import load
        model, cfg, _ = load(args.ckpt, device='cpu')
        D, H, L = cfg['d_model'], cfg['n_head'], cfg['n_layer']
    else:
        D, H, L = 16, 2, 1                     # 走查用超小模型, 数字看得清
        model = MiniTransformer(D, H, L, block_size=32, dropout=0.0)
    model.eval()

    s = args.text.lower()
    ids = encode(s)
    x = torch.tensor([ids])
    T = len(ids)

    sep('第 0 步  分词: 字符 -> token id')
    print(f'  输入文本   : "{s}"')
    print(f'  token ids  : {ids}')
    print(f'  还原       : "{decode(ids)}"')
    print(f'  词表大小 V = {VOCAB_SIZE}, 序列长 T = {T}')

    sep('第 1 步  查表嵌入: one-hot(T x 27) @ E(27 x D) = (T x D)')
    E = model.tok_emb.weight.detach()
    oh = one_hot(ids)
    emb_matmul = oh @ E
    emb_lookup = model.tok_emb(x)[0].detach()
    print(f'  one-hot 形状 : {tuple(oh.shape)}   每行只有 1 个 1')
    print(f'  嵌入矩阵 E   : {tuple(E.shape)}   <- 27 个 token 各占一行')
    print(f'  相乘结果     : {tuple(emb_matmul.shape)}')
    print(f'  与查表一致?  : {torch.allclose(emb_matmul, emb_lookup, atol=1e-6)}')
    for ch, v in zip(s, emb_lookup):
        print(f"    '{ch if ch != ' ' else '_'}' -> {brief(v)}")

    sep('第 2 步  加位置编码: x = token_emb + pos_emb')
    pos = model.pos_emb(torch.arange(T)).detach()
    h = emb_lookup + pos
    print(f'  位置嵌入 P   : {tuple(model.pos_emb.weight.shape)} 取前 {T} 行')
    print(f'  位置 0 向量  : {brief(pos[0])}')
    print(f'  相加后 x     : {tuple(h.shape)}')
    print('  注意: 同一个字母出现在不同位置, 加完位置编码后向量就不同了 ->')
    same = [i for i, c in enumerate(s) if s.count(c) > 1]
    if len(same) >= 2:
        i, j = same[0], same[1]
        print(f"    位置 {i} 的 '{s[i]}': {brief(h[i], 4)}")
        print(f"    位置 {j} 的 '{s[j]}': {brief(h[j], 4)}")

    sep('第 3 步  自注意力: Q K V -> 打分 -> 掩码 -> softmax -> 加权求和')
    blk = model.blocks[0]
    hn = blk.ln1(h.unsqueeze(0))                       # LayerNorm 先做归一化
    q, k, v = blk.attn.qkv(hn).split(D, dim=2)
    d_head = D // H
    qh = q.view(1, T, H, d_head).transpose(1, 2)
    kh = k.view(1, T, H, d_head).transpose(1, 2)
    vh = v.view(1, T, H, d_head).transpose(1, 2)
    scores = (qh @ kh.transpose(-2, -1)) / math.sqrt(d_head)
    masked = scores.masked_fill(blk.attn.mask[:, :, :T, :T] == 0, float('-inf'))
    att = F.softmax(masked, dim=-1)
    ctx = att @ vh

    print(f'  Q/K/V 形状(拆头后) : {tuple(qh.shape)} = (B, head={H}, T={T}, d_head={d_head})')
    lab = [c if c != ' ' else '_' for c in s]
    print(f'\n  [head 0] 原始打分 QK^T/sqrt(d_head):')
    print('        ' + '  '.join(f'{c:>6}' for c in lab))
    for i in range(T):
        print(f'   {lab[i]:>3} ' + '  '.join(f'{scores[0,0,i,j]:>6.2f}' for j in range(T)))
    print(f'\n  [head 0] 因果掩码后 softmax (每行和=1, 上三角被屏蔽为 0):')
    print('        ' + '  '.join(f'{c:>6}' for c in lab))
    for i in range(T):
        print(f'   {lab[i]:>3} ' + '  '.join(f'{att[0,0,i,j]:>6.2f}' for j in range(T)))
    print(f'\n  加权求和后 context : {tuple(ctx.shape)}, 拼回 {tuple((1, T, D))}')

    sep('第 4 步  残差 + FFN, 逐层堆叠')
    hh = h.unsqueeze(0)
    for li, b in enumerate(model.blocks):
        a, _ = b.attn(b.ln1(hh))
        hh1 = hh + a
        hh = hh1 + b.ffn(b.ln2(hh1))
        print(f'  第 {li} 层输出 : {tuple(hh.shape)}  均值 {hh.mean():+.4f}  标准差 {hh.std():.4f}')
    print('  残差连接的作用: 让原始信息与梯度都能直通到顶层, 深层网络才训得动')

    sep('第 5 步  输出头: (T x D) @ W_out(D x 27) -> logits(T x 27) -> softmax')
    logits, _, _ = model(x)
    probs = F.softmax(logits[0], dim=-1)
    print(f'  logits 形状 : {tuple(logits.shape)}  (每个位置对 27 个 token 打分)')
    for i in range(T):
        p, idx = probs[i].topk(3)
        top = ' '.join(f"'{ITOS[j] if ITOS[j] != ' ' else '_'}'={x:.2f}" for x, j in zip(p, idx))
        nxt = s[i + 1] if i + 1 < T else '?'
        print(f"    看到 \"{s[:i+1]}\" -> 预测下一个: {top}   (真实答案 '{nxt}')")

    sep('第 6 步  损失 + 反向传播 + 一次参数更新')
    y = torch.tensor([ids[1:] + [0]])                 # 目标 = 输入右移一位
    model.train()
    _, loss, _ = model(x, y)
    print(f'  交叉熵 loss  : {loss.item():.4f}   (随机模型应约 ln27={math.log(27):.4f})')
    print(f'  困惑度 ppl   : {math.exp(loss.item()):.2f}  (相当于在 27 个字符里"犹豫"这么多个)')

    model.zero_grad()
    loss.backward()
    print('\n  反向传播后各参数的梯度范数(越大 = 这一步被改动越多):')
    for n, p in model.named_parameters():
        if p.grad is not None:
            print(f'    {n:<38} 形状 {str(tuple(p.shape)):<12} |grad| = {p.grad.norm():.5f}')

    # 嵌入矩阵每一行的梯度: 直接反映"这一步谁被学到了"
    grow = model.tok_emb.weight.grad.norm(dim=1)
    print('\n  嵌入矩阵 E 每行(每个 token)的梯度范数, 从大到小:')
    for i in grow.argsort(descending=True)[:8]:
        c = ITOS[i] if ITOS[i] != ' ' else '_'
        mark = '  <- 出现在本批输入中' if i.item() in ids else ''
        print(f"    '{c}'  |grad| = {grow[i]:.5f}{mark}")
    print('  说明: 本批出现过的 token 同时走"输入查表"和"输出打分"两条路径, 梯度最大;')
    print('        其余 token 因为本模型做了权重绑定(输出头复用 E), 也会收到一份')
    print('        "把错误候选的分数压低"的梯度, 但通常小得多。')

    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    opt.step()                              # 真正修改参数
    _, loss2, _ = model(x, y)
    print(f'\n  更新一步后 loss: {loss.item():.4f} -> {loss2.item():.4f} '
          f'({"下降" if loss2 < loss else "上升"} {abs(loss2.item()-loss.item()):.4f})')
    print('  把这个循环重复几千次, 就是 train.py 做的事。')


if __name__ == '__main__':
    main()
