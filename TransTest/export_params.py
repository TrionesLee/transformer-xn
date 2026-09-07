"""
把训练完的模型参数导出成人类可读的 JSON —— 用它可以脱离 PyTorch 手工算出概率。

导出内容:
    meta     词表、结构配置、训练信息
    formula  每一步的数学公式(手算时照着套)
    tensors  全部参数矩阵，附带形状与用途说明

配套 handcalc.py 只用标准库读这个 JSON，就能把概率一步步算出来。
"""

import json
import os
import torch

from data import ITOS, VOCAB_SIZE


# 每个参数张量的用途说明（手算时知道它是干什么的）
DESC = {
    'tok_emb.weight':   '嵌入矩阵 E (27, D): 第 i 行 = token i 的向量。也复用作输出头(权重绑定)',
    'pos_emb.weight':   '位置嵌入 P (block_size, D): 第 t 行 = 位置 t 的向量',
    'ln1.weight':       '注意力前 LayerNorm 的缩放 γ (D,)',
    'ln1.bias':         '注意力前 LayerNorm 的平移 β (D,)',
    'attn.qkv.weight':  'Q/K/V 三个投影拼在一起 (3D, D): 前 D 行算 Q，中间 D 行算 K，后 D 行算 V',
    'attn.proj.weight': '多头拼接后的输出投影 W_O (D, D)',
    'attn.proj.bias':   '输出投影偏置 b_O (D,)',
    'ln2.weight':       'FFN 前 LayerNorm 的缩放 γ (D,)',
    'ln2.bias':         'FFN 前 LayerNorm 的平移 β (D,)',
    'ffn.net.0.weight': 'FFN 第一层 W1 (4D, D): 升维',
    'ffn.net.0.bias':   'FFN 第一层偏置 b1 (4D,)',
    'ffn.net.2.weight': 'FFN 第二层 W2 (D, 4D): 降维回 D',
    'ffn.net.2.bias':   'FFN 第二层偏置 b2 (D,)',
    'ln_f.weight':      '输出前最后一次 LayerNorm 的缩放 γ (D,)',
    'ln_f.bias':        '输出前最后一次 LayerNorm 的平移 β (D,)',
}

FORMULA = [
    "1. 分词      : 字符 c -> token id  (' '=0, 'a'=1, ..., 'z'=26)",
    "2. 嵌入      : x[t] = E[id_t] + P[t]                       # 两个向量逐元素相加",
    "3. LayerNorm : n = (x - mean(x)) / sqrt(var(x) + 1e-5) * γ + β   # var 用有偏估计(除以 D)",
    "4. QKV       : q = W_q·n , k = W_k·n , v = W_v·n           # W_q/W_k/W_v 是 qkv.weight 的三段",
    "5. 注意力分数: s[t][j] = (q[t]·k[j]) / sqrt(d_head)  , 仅 j <= t (因果掩码)",
    "6. softmax   : a[t][j] = exp(s[t][j] - max_j s) / Σ_j exp(s[t][j] - max_j s)",
    "7. 加权求和  : ctx[t] = Σ_j a[t][j] * v[j]                 # 多头则各头算完再拼接",
    "8. 输出投影  : x = x + (W_O·ctx + b_O)                     # 残差相加",
    "9. FFN       : h = GELU(W1·LN2(x) + b1) ; x = x + (W2·h + b2)",
    "   GELU(u)   = 0.5 * u * (1 + erf(u / sqrt(2)))",
    "10. 最终归一 : z = LayerNorm_f(x[最后一个位置])",
    "11. 打分     : logit[i] = E[i] · z        # 权重绑定，输出头就是嵌入矩阵",
    "12. 概率     : P(下一个字符 = i) = exp(logit[i] - max) / Σ_k exp(logit[k] - max)",
]


def _round(t, nd=6):
    """张量 -> 嵌套 list，保留 nd 位小数，便于阅读和手算"""
    if t.dim() == 0:
        return round(float(t), nd)
    return [_round(s, nd) for s in t]


def export_json(model, path='params.json', nd=6, extra=None):
    """把 model 的全部参数写成 JSON。返回统计信息。"""
    cfg = dict(
        vocab_size=VOCAB_SIZE,
        d_model=model.tok_emb.weight.shape[1],
        n_layer=len(model.blocks),
        n_head=model.blocks[0].attn.n_head,
        d_head=model.blocks[0].attn.d_head,
        block_size=model.block_size,
        eps=1e-5,
        weight_tying=True,
    )
    tensors, descs, n_scalars = {}, {}, 0
    for name, p in model.state_dict().items():
        tensors[name] = _round(p.detach().cpu(), nd)
        n_scalars += p.numel()
        key = next((k for k in DESC if name.endswith(k)), None)
        descs[name] = (DESC[key] if key else '') + f'   形状 {tuple(p.shape)}'

    doc = {
        'meta': {
            'note': '27-token 迷你 Transformer 的全部参数。配合 handcalc.py 可脱离 PyTorch 手算概率。',
            'vocab': {str(i): (' ' if c == ' ' else c) for i, c in enumerate(ITOS)},
            'vocab_note': "token 0 是空格; 1..26 依次是 a..z",
            'config': cfg,
            'training': extra or {},
        },
        'formula': FORMULA,
        'tensor_desc': descs,
        'tensors': tensors,
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    return dict(kb=os.path.getsize(path) / 1024, n_tensors=len(tensors),
                n_scalars=n_scalars, config=cfg)


def main():
    """独立使用: 从 ckpt.pt 导出 params.json"""
    import argparse
    from inspect_model import load
    ap = argparse.ArgumentParser(description='把 ckpt.pt 的参数导出为可手算的 JSON')
    ap.add_argument('--ckpt', default='ckpt.pt')
    ap.add_argument('--out', default='params.json')
    ap.add_argument('--nd', type=int, default=6, help='保留几位小数')
    ap.add_argument('--print', dest='show', action='store_true',
                    help='同时把参数打印到屏幕(仅适合 --tiny 规模的模型)')
    args = ap.parse_args()

    model, cfg, device = load(args.ckpt, device='cpu')
    info = export_json(model, args.out, args.nd)
    print(f'\n已导出 {args.out}')
    print(f'  结构      : {cfg["n_layer"]} 层 x {cfg["n_head"]} 头, d_model={cfg["d_model"]}, '
          f'上下文 {cfg["block_size"]}')
    print(f'  张量个数  : {info["n_tensors"]}')
    print(f'  参数个数  : {info["n_scalars"]:,}')
    print(f'  文件大小  : {info["kb"]:.0f} KB')
    print('\n手算公式:')
    for line in FORMULA:
        print('  ' + line)
    print(f'\n验证手算结果:  python3 handcalc.py --params {args.out} --prompt "the c"')

    if args.show:
        import json as _j
        doc = _j.load(open(args.out, encoding='utf-8'))
        print('\n全部参数:')
        for n, t in doc['tensors'].items():
            print(f'\n{n}  —— {doc["tensor_desc"][n]}')
            print('  ' + _j.dumps(t))
    return 0


if __name__ == '__main__':
    main()
