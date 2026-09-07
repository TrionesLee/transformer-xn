"""
手算下一个字符的概率 —— 只用 Python 标准库（math/json），完全不导入 PyTorch。

作用: 证明导出的 params.json 里就是全部信息, 拿计算器照着算也能得到同样的概率;
      每一步都打印中间数值和用到的公式, 可以逐行核对。

    python3 handcalc.py --prompt "the c"              # 完整逐步演算
    python3 handcalc.py --prompt "the c" --brief      # 只看结果
    python3 handcalc.py --prompt "q" --verify         # 与 PyTorch 结果对拍
"""

import argparse
import json
import math

EPS_DEFAULT = 1e-5


# ---------------------------------------------------------------- 基础线性代数
def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def matvec(M, v):
    """矩阵(行优先) 乘 向量: 结果[i] = M[i] · v"""
    return [dot(row, v) for row in M]


def vadd(a, b):
    return [x + y for x, y in zip(a, b)]


def layer_norm(x, gamma, beta, eps=EPS_DEFAULT):
    """LayerNorm: 先把向量标准化成均值0方差1, 再用 γ 缩放、β 平移"""
    n = len(x)
    mean = sum(x) / n
    var = sum((v - mean) ** 2 for v in x) / n        # 有偏方差(除以 n)
    inv = 1.0 / math.sqrt(var + eps)
    return [(v - mean) * inv * g + b for v, g, b in zip(x, gamma, beta)], mean, var


def softmax(xs):
    m = max(xs)
    e = [math.exp(v - m) for v in xs]                 # 减最大值防溢出
    s = sum(e)
    return [v / s for v in e]


def gelu(u):
    """PyTorch nn.GELU 默认用精确版: 0.5u(1+erf(u/√2))"""
    return 0.5 * u * (1.0 + math.erf(u / math.sqrt(2.0)))


# ---------------------------------------------------------------- 前向
def forward(P, ids, trace=None):
    """
    用纯 Python 跑一遍前向, 返回最后一个位置对 27 个 token 的概率。
    trace 不为 None 时, 把中间结果塞进去供打印。
    """
    T_ = P['tensors']
    cfg = P['meta']['config']
    D, H, dh = cfg['d_model'], cfg['n_head'], cfg['d_head']
    eps, L = cfg.get('eps', EPS_DEFAULT), cfg['n_layer']
    E, Pe = T_['tok_emb.weight'], T_['pos_emb.weight']
    T = len(ids)

    # ---- 第 2 步: 嵌入 + 位置编码 ----
    xs = [vadd(E[i], Pe[t]) for t, i in enumerate(ids)]
    if trace is not None:
        trace['emb'] = [list(E[i]) for i in ids]
        trace['pos'] = [list(Pe[t]) for t in range(T)]
        trace['x0'] = [list(v) for v in xs]

    for L_i in range(L):
        pre = f'blocks.{L_i}.'
        g1, b1 = T_[pre + 'ln1.weight'], T_[pre + 'ln1.bias']
        Wqkv = T_[pre + 'attn.qkv.weight']
        Wo, bo = T_[pre + 'attn.proj.weight'], T_[pre + 'attn.proj.bias']
        g2, b2 = T_[pre + 'ln2.weight'], T_[pre + 'ln2.bias']
        W1, c1 = T_[pre + 'ffn.net.0.weight'], T_[pre + 'ffn.net.0.bias']
        W2, c2 = T_[pre + 'ffn.net.2.weight'], T_[pre + 'ffn.net.2.bias']

        # ---- 第 3~4 步: LayerNorm 再算 Q K V ----
        ns, stats = [], []
        for v in xs:
            n_, mu, va = layer_norm(v, g1, b1, eps)
            ns.append(n_)
            stats.append((mu, va))
        qkv = [matvec(Wqkv, n_) for n_ in ns]          # 每个位置得到长度 3D 的向量
        q = [z[0:D] for z in qkv]
        k = [z[D:2 * D] for z in qkv]
        v_ = [z[2 * D:3 * D] for z in qkv]

        # ---- 第 5~7 步: 逐头算注意力 ----
        ctx = [[0.0] * D for _ in range(T)]
        att_dump = []
        scale = 1.0 / math.sqrt(dh)
        for h in range(H):
            s0, s1 = h * dh, (h + 1) * dh
            head_att = []
            for t in range(T):
                qi = q[t][s0:s1]
                scores = [dot(qi, k[j][s0:s1]) * scale for j in range(t + 1)]  # 因果掩码
                a = softmax(scores)
                head_att.append(a)
                for j in range(t + 1):
                    for d in range(dh):
                        ctx[t][s0 + d] += a[j] * v_[j][s0 + d]
            att_dump.append(head_att)

        # ---- 第 8 步: 输出投影 + 残差 ----
        xs = [vadd(x, vadd(matvec(Wo, c), bo)) for x, c in zip(xs, ctx)]

        # ---- 第 9 步: LayerNorm + FFN + 残差 ----
        ffn_dump = []
        new = []
        for x in xs:
            n2, _, _ = layer_norm(x, g2, b2, eps)
            hid = [gelu(u) for u in vadd(matvec(W1, n2), c1)]
            out = vadd(matvec(W2, hid), c2)
            ffn_dump.append((n2, hid, out))
            new.append(vadd(x, out))
        xs = new

        if trace is not None and L_i == 0:
            trace.update(ln1=ns, ln1_stats=stats, q=q, k=k, v=v_,
                         att=att_dump, ctx=ctx, after_attn=[list(z) for z in xs],
                         ffn=ffn_dump)

    # ---- 第 10~12 步: 最终 LayerNorm -> 打分 -> softmax ----
    gf, bf = T_['ln_f.weight'], T_['ln_f.bias']
    z, mu_f, va_f = layer_norm(xs[-1], gf, bf, eps)
    logits = [dot(E[i], z) for i in range(cfg['vocab_size'])]   # 权重绑定: 输出头就是 E
    probs = softmax(logits)
    if trace is not None:
        trace.update(z=z, lnf_stats=(mu_f, va_f), logits=logits, probs=probs)
    return probs


# ---------------------------------------------------------------- 打印
def fmt(v, n=6, nd=4):
    body = ', '.join(f'{x:+.{nd}f}' for x in v[:n])
    return f'[{body}{", ..." if len(v) > n else ""}]'


def show(P, prompt, ids, trace, probs, ncol=6, topk=8):
    cfg = P['meta']['config']
    D, H, dh = cfg['d_model'], cfg['n_head'], cfg['d_head']
    itos = P['meta']['vocab']
    ch = lambda i: '␣' if itos[str(i)] == ' ' else itos[str(i)]
    T = len(ids)
    line = lambda t: print('─' * 78) if t is None else (print('─' * 78), print(t))

    print('=' * 78)
    print(f'手算 P(下一个字符 | "{prompt}")   —— 纯 Python，未使用 PyTorch')
    print('=' * 78)
    print(f'模型: {cfg["n_layer"]} 层 x {H} 头, d_model={D}, 每头维度 d_head={dh}, '
          f'词表 {cfg["vocab_size"]}')
    print(f'(向量只显示前 {ncol} 个分量，完整数值都在 params.json 里)')

    print('\n【第 1 步】分词: 字符 -> token id')
    print(f'  "{prompt}"  ->  {ids}')
    print('  ' + '  '.join(f"'{ch(i)}'={i}" for i in ids))

    print('\n【第 2 步】嵌入: x[t] = E[id_t] + P[t]   (查表得到的两个向量逐元素相加)')
    for t, i in enumerate(ids):
        print(f'  位置{t} \'{ch(i)}\':  E[{i:>2}] = {fmt(trace["emb"][t], ncol)}')
        print(f'            + P[{t:>2}] = {fmt(trace["pos"][t], ncol)}')
        print(f'            = x[{t:>2}]  = {fmt(trace["x0"][t], ncol)}')

    last = T - 1
    print(f'\n【第 3 步】LayerNorm: n = (x - 均值) / sqrt(方差 + {cfg.get("eps", 1e-5)}) * γ + β')
    mu, va = trace['ln1_stats'][last]
    print(f'  只看最后一个位置 {last} (\'{ch(ids[last])}\'):')
    print(f'    均值 = {mu:+.6f}   方差 = {va:.6f}   1/sqrt(方差+eps) = '
          f'{1 / math.sqrt(va + cfg.get("eps", 1e-5)):.6f}')
    print(f'    归一化后 n = {fmt(trace["ln1"][last], ncol)}')

    print('\n【第 4 步】Q/K/V: 用 qkv.weight 的三段分别乘 n')
    print(f'    q[{last}] = {fmt(trace["q"][last], ncol)}')
    print(f'    k[{last}] = {fmt(trace["k"][last], ncol)}')
    print(f'    v[{last}] = {fmt(trace["v"][last], ncol)}')

    print(f'\n【第 5~7 步】注意力: 分数 = q·k / sqrt({dh}) = q·k * {1/math.sqrt(dh):.6f}，'
          f'softmax 后加权求和 v')
    for h in range(H):
        a = trace['att'][h][last]
        sc = [dot(trace['q'][last][h*dh:(h+1)*dh], trace['k'][j][h*dh:(h+1)*dh])
              / math.sqrt(dh) for j in range(T)]
        print(f'  [头 {h}] 最后位置对各位置的原始分数:')
        print('     ' + '  '.join(f"{ch(ids[j])}:{sc[j]:+.3f}" for j in range(T)))
        print(f'  [头 {h}] softmax 后的注意力权重(和为 {sum(a):.4f}):')
        print('     ' + '  '.join(f"{ch(ids[j])}:{a[j]:.3f}" for j in range(T)))
    print(f'  加权求和得 ctx = {fmt(trace["ctx"][last], ncol)}')

    print('\n【第 8 步】输出投影 + 残差: x = x + (W_O·ctx + b_O)')
    print(f'    x = {fmt(trace["after_attn"][last], ncol)}')

    n2, hid, out = trace['ffn'][last]
    print('\n【第 9 步】FFN: h = GELU(W1·LN2(x) + b1);  x = x + (W2·h + b2)')
    print(f'    LN2(x)  = {fmt(n2, ncol)}')
    print(f'    GELU 后 = {fmt(hid, ncol)}   (共 {len(hid)} 个隐藏单元)')
    print(f'    W2·h+b2 = {fmt(out, ncol)}')

    print('\n【第 10 步】最后一次 LayerNorm')
    mu_f, va_f = trace['lnf_stats']
    print(f'    均值 = {mu_f:+.6f}  方差 = {va_f:.6f}')
    print(f'    z = {fmt(trace["z"], ncol)}')

    print('\n【第 11 步】打分: logit[i] = E[i] · z   (权重绑定，输出头就是嵌入矩阵 E)')
    lg = trace['logits']
    order = sorted(range(len(lg)), key=lambda i: -lg[i])[:topk]
    for i in order:
        print(f"    logit['{ch(i)}'] = E[{i:>2}] · z = {lg[i]:+.6f}")

    print('\n【第 12 步】softmax 得到概率')
    m = max(lg)
    denom = sum(math.exp(x - m) for x in lg)
    print(f'    最大 logit = {m:.6f}，分母 Σexp(logit - 最大值) = {denom:.6f}')
    for i in order:
        p = probs[i]
        print(f"    P('{ch(i)}') = exp({lg[i]:+.4f} - {m:.4f}) / {denom:.4f} "
              f"= {p:.6f} = {p*100:5.2f}%  {'█' * int(p * 40)}")
    print(f'\n  27 个概率之和 = {sum(probs):.6f}  (应为 1)')


def main():
    ap = argparse.ArgumentParser(description='用导出的参数手算下一个字符的概率')
    ap.add_argument('--params', default='params.json', help='train.py 导出的参数文件')
    ap.add_argument('--prompt', default='the c', help='提示词')
    ap.add_argument('--ncol', type=int, default=6, help='向量显示前几个分量')
    ap.add_argument('--topk', type=int, default=8)
    ap.add_argument('--brief', action='store_true', help='只输出结果，不打印推导过程')
    ap.add_argument('--verify', action='store_true', help='与 PyTorch 的结果对拍')
    ap.add_argument('--ckpt', default='ckpt.pt', help='对拍用的模型文件')
    args = ap.parse_args()

    with open(args.params, encoding='utf-8') as f:
        P = json.load(f)
    cfg = P['meta']['config']

    prompt = ''.join(c if 'a' <= c <= 'z' else ' ' for c in args.prompt.lower())
    stoi = {v: int(k) for k, v in P['meta']['vocab'].items()}
    ids = [stoi.get(c, 0) for c in prompt][-cfg['block_size']:]
    if not ids:
        ids = [0]

    trace = {}
    probs = forward(P, ids, trace)

    if args.brief:
        itos = P['meta']['vocab']
        order = sorted(range(len(probs)), key=lambda i: -probs[i])[:args.topk]
        print(f'P(下一个字符 | "{prompt}"):')
        for i in order:
            c = '␣' if itos[str(i)] == ' ' else itos[str(i)]
            print(f"  '{c}'  {probs[i]*100:6.2f}%  {'█' * int(probs[i]*40)}")
    else:
        show(P, prompt, ids, trace, probs, args.ncol, args.topk)

    if args.verify:
        print('\n' + '=' * 78)
        print('与 PyTorch 对拍')
        print('=' * 78)
        import torch
        from inspect_model import load
        from data import encode
        model, _, _ = load(args.ckpt, device='cpu')
        with torch.no_grad():
            logits, _, _ = model(torch.tensor([ids]))
            ref = torch.softmax(logits[0, -1], dim=-1).tolist()
        diff = max(abs(a - b) for a, b in zip(probs, ref))
        order = sorted(range(len(probs)), key=lambda i: -ref[i])[:args.topk]
        itos = P['meta']['vocab']
        print(f'{"字符":>6} {"手算":>12} {"PyTorch":>12} {"差":>12}')
        for i in order:
            c = '␣' if itos[str(i)] == ' ' else itos[str(i)]
            print(f'{c:>6} {probs[i]:>12.8f} {ref[i]:>12.8f} {abs(probs[i]-ref[i]):>12.2e}')
        print(f'\n27 个概率的最大绝对误差 = {diff:.3e}')
        print('结论: ' + ('一致（误差仅来自导出时的小数截断）' if diff < 1e-3
                        else '不一致，请检查参数导出'))
    return 0


if __name__ == '__main__':
    main()
