"""终端可视化: 不依赖 matplotlib, 用灰度块字符画热力图。"""

import torch
from data import ITOS, VOCAB_SIZE

SHADES = ' .:-=+*#%@'          # 由浅到深的 10 级灰度


def _label(i):
    return '_' if ITOS[i] == ' ' else ITOS[i]


def heatmap(mat, row_labels=None, col_labels=None, title='', vmin=None, vmax=None,
            cell=1, robust=False):
    """把 2D 张量画成灰度块热力图; robust=True 时按 mean±2std 裁剪, 避免少数极值吃掉对比度"""
    m = mat.detach().float().cpu()
    if robust and vmin is None and vmax is None:
        mu, sd = m.mean().item(), m.std().item()
        vmin, vmax = mu - 2 * sd, mu + 2 * sd
    lo = m.min().item() if vmin is None else vmin
    hi = m.max().item() if vmax is None else vmax
    rng = (hi - lo) or 1e-9
    rows, cols = m.shape

    out = []
    if title:
        out.append(f'\n{title}   [范围 {lo:+.3f} ~ {hi:+.3f}]')
    if col_labels is not None:
        out.append('    ' + ''.join(str(c).rjust(cell) for c in col_labels))
    for r in range(rows):
        lab = (str(row_labels[r]) if row_labels is not None else str(r)).rjust(3)
        line = ''.join(
            SHADES[max(0, min(9, int((m[r, c].item() - lo) / rng * 9.999)))] * cell
            for c in range(cols))
        out.append(f'{lab} {line}')
    return '\n'.join(out)


def show_embedding(model, title='嵌入矩阵 E (27 x d_model): 每行 = 一个 token 在向量空间中的坐标'):
    E = model.tok_emb.weight
    return heatmap(E, row_labels=[_label(i) for i in range(VOCAB_SIZE)],
                   title=title, robust=True)


def show_similarity(model, topk=4):
    """token 之间的余弦相似度: 训练后语义/用法相近的字母(如元音)会靠得更近"""
    E = torch.nn.functional.normalize(model.tok_emb.weight.detach(), dim=1)
    sim = E @ E.T
    txt = heatmap(sim, row_labels=[_label(i) for i in range(VOCAB_SIZE)],
                  col_labels=[_label(i) for i in range(VOCAB_SIZE)],
                  title='token 余弦相似度矩阵 (对角线=自己=1)', vmin=-1, vmax=1)
    lines = [txt, '\n每个 token 的最近邻:']
    s = sim.clone()
    s.fill_diagonal_(-2)
    v, idx = s.topk(topk, dim=1)
    for i in range(VOCAB_SIZE):
        nb = ' '.join(f'{_label(j.item())}({x:+.2f})' for j, x in zip(idx[i], v[i]))
        lines.append(f"  '{_label(i)}' -> {nb}")
    return '\n'.join(lines)


def show_attention(model, ids, device='cpu', layer=0, head=0):
    """展示某层某头的注意力矩阵: 第 i 行 = 生成第 i 个位置时, 对前面各位置的关注权重"""
    x = torch.tensor([ids], device=device)
    with torch.no_grad():
        model.eval()
        _, _, attns = model(x, return_attn=True)
        model.train()
    att = attns[layer][0, head]                       # (T, T)
    labels = [_label(i) for i in ids]
    return heatmap(att, row_labels=labels, col_labels=labels,
                   title=f'注意力权重  第 {layer} 层 / 第 {head} 头  '
                         f'(行=查询位置, 列=被关注位置, 上三角因掩码恒为 0)',
                   vmin=0, vmax=1)


def show_next_char_dist(model, prefix, device='cpu', topk=8):
    """给定前缀, 看模型对下一个字符的概率分布 —— 最直观的"学到了什么" """
    from data import encode
    ids = encode(prefix)[-model.block_size:]
    x = torch.tensor([ids], device=device)
    with torch.no_grad():
        model.eval()
        logits, _, _ = model(x)
        model.train()
    probs = torch.softmax(logits[0, -1], dim=-1)
    v, idx = probs.topk(topk)
    bars = []
    for p, i in zip(v.tolist(), idx.tolist()):
        bars.append(f"    '{_label(i)}'  {p*100:5.1f}%  {'█' * int(p * 40)}")
    return f'前缀 "{prefix}" -> 下一个字符预测:\n' + '\n'.join(bars)


def loss_curve(history, width=64, height=12):
    """ASCII 折线图: 训练损失下降曲线"""
    if len(history) < 2:
        return ''
    steps = [h[0] for h in history]
    vals = [h[1] for h in history]
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1e-9
    # 重采样到 width 列
    grid = [[' '] * width for _ in range(height)]
    for c in range(width):
        i = int(c / (width - 1) * (len(vals) - 1))
        r = height - 1 - int((vals[i] - lo) / rng * (height - 1))
        grid[r][c] = '*'
    lines = [f'\n训练损失曲线 (纵轴 loss, 横轴 step {steps[0]} -> {steps[-1]})']
    for r, row in enumerate(grid):
        y = hi - r / (height - 1) * rng
        lines.append(f'{y:6.3f} |' + ''.join(row))
    lines.append('       +' + '-' * width)
    return '\n'.join(lines)
