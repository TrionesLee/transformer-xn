"""
加载训练好的 ckpt.pt, 免重训地探查模型内部。

用法:
    python3 inspect_model.py                       # 全部分析
    python3 inspect_model.py --prefix "the qu"     # 看某个前缀的预测 + 注意力
    python3 inspect_model.py --gen 400 --temp 0.7  # 只生成文本
"""

import argparse
import torch

from data import encode, decode, one_hot
from model import MiniTransformer
import visualize as viz


def load(path='ckpt.pt', device=None):
    """加载 ckpt; device 为 None 时自动识别 CPU/GPU"""
    ck = torch.load(path, map_location='cpu', weights_only=False)
    cfg = ck.get('config') or {k: ck['args'][k] for k in
                               ('d_model', 'n_head', 'n_layer', 'block_size', 'dropout')}
    if device is None:
        from train import pick_device
        device = pick_device('auto')
    m = MiniTransformer(**cfg)
    m.load_state_dict(ck['model'])
    return m.to(device).eval(), cfg, device


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', default='ckpt.pt')
    p.add_argument('--prefix', default='the quick brown')
    p.add_argument('--gen', type=int, default=200)
    p.add_argument('--temp', type=float, default=0.8)
    p.add_argument('--layer', type=int, default=None)
    p.add_argument('--head', type=int, default=0)
    args = p.parse_args()

    model, cfg, device = load(args.ckpt)
    print(f"模型: {cfg['n_layer']} 层 x {cfg['n_head']} 头, d_model={cfg['d_model']}, "
          f"参数量 {model.n_params():,}\n")

    # 演示: 查表 == one-hot 乘嵌入矩阵
    ids = encode('cat')
    E = model.tok_emb.weight.detach().cpu()
    lookup = E[torch.tensor(ids)]
    matmul = one_hot(ids) @ E
    print('【嵌入 = one-hot × E】')
    print(f'  "cat" -> ids {ids}')
    print(f'  查表结果与 one-hot@E 是否一致: {torch.allclose(lookup, matmul, atol=1e-6)}')
    print(f'  每个字符得到一个 {E.shape[1]} 维向量, 前 8 维: ')
    for ch, v in zip('cat', lookup):
        print(f"    '{ch}' -> [{', '.join(f'{x:+.3f}' for x in v[:8])} ...]")

    print(viz.show_embedding(model))
    print(viz.show_similarity(model))

    ids = encode(args.prefix)[:cfg['block_size']]
    layers = range(cfg['n_layer']) if args.layer is None else [args.layer]
    for L in layers:
        print(viz.show_attention(model, ids, device, layer=L, head=args.head))

    print('\n【下一个字符预测】')
    print(viz.show_next_char_dist(model, args.prefix, device))

    x0 = torch.zeros((1, 1), dtype=torch.long, device=device)
    out = decode(model.generate(x0, args.gen, temperature=args.temp, top_k=8)[0])
    print(f'\n【生成 (temperature={args.temp})】\n{out}')


if __name__ == '__main__':
    main()
