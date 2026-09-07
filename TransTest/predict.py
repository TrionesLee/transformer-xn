"""
第二部分：预测
    - 加载 train.py 训练出的模型参数 (ckpt.pt)
    - 自动识别 CPU / GPU
    - 输入提示词(prompt), 自回归续写出完整的句子

用法:
    python3 predict.py --prompt "the qu"              # 单次预测
    python3 predict.py                                # 交互模式, 反复输入提示词
    python3 predict.py --prompt "peo" --words 20      # 续写 20 个完整单词
    python3 predict.py --prompt "the" --n 5           # 一次给出 5 个候选续写
    python3 predict.py --prompt "wat" --show-probs    # 同时显示下一个字符的概率分布
    python3 predict.py --prompt "the" --temp 0.2      # 低温度 = 更保守/更确定
"""

import argparse
import math
import torch
import torch.nn.functional as F

from data import encode, decode, ITOS, VOCAB_SIZE
from model import MiniTransformer
from train import pick_device


# ------------------------------------------------------------------ 加载模型
def load_model(path, device):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    cfg = ck.get('config') or {k: ck['args'][k] for k in
                               ('d_model', 'n_head', 'n_layer', 'block_size', 'dropout')}
    model = MiniTransformer(**cfg)
    model.load_state_dict(ck['model'])
    model.to(device).eval()
    return model, cfg, ck.get('stats', {})


def clean_prompt(s):
    """提示词也要折叠到 27 个 token 的范围内"""
    s = ''.join(c if 'a' <= c <= 'z' else ' ' for c in s.lower())
    return ' '.join(s.split())


# ------------------------------------------------------------------ 生成
@torch.no_grad()
def complete(model, prompt, device, words=12, temperature=0.8, top_k=8, max_chars=400):
    """
    从提示词续写, 直到凑满 `words` 个完整单词为止 —— 保证不会在半个单词处截断。

    每一步:  取最近 block_size 个字符 -> 前向 -> 取最后位置的 logits
             -> 温度缩放 -> top-k 过滤 -> softmax -> 按概率采样一个字符 -> 拼回去
    """
    ids = encode(prompt) if prompt else [0]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    n_words, generated = 0, 0
    # 提示词末尾若不在词中间, 记一次边界, 避免把提示词自身的词算进去
    while generated < max_chars:
        cond = idx[:, -model.block_size:]
        logits, _, _ = model(cond)
        logits = logits[0, -1] / max(temperature, 1e-6)
        if top_k:
            v, _ = torch.topk(logits, min(top_k, VOCAB_SIZE))
            logits = logits.masked_fill(logits < v[-1], -float('inf'))
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1)
        idx = torch.cat([idx, nxt.view(1, 1)], dim=1)
        generated += 1
        if nxt.item() == 0:                       # 采到空格 = 一个单词写完了
            n_words += 1
            if n_words >= words:
                break
    out = decode(idx[0])
    return ' '.join(out.split())                  # 规整多余空格


@torch.no_grad()
def next_char_probs(model, prompt, device, topk=8):
    ids = encode(prompt)[-model.block_size:] or [0]
    logits, _, _ = model(torch.tensor([ids], device=device))
    probs = F.softmax(logits[0, -1], dim=-1)
    v, i = probs.topk(topk)
    return [(ITOS[j] if ITOS[j] != ' ' else '␣', p) for p, j in zip(v.tolist(), i.tolist())]


@torch.no_grad()
def score(model, text, device):
    """给一段文本打分: 平均每字符交叉熵 + 困惑度, 衡量模型觉得它有多"像英语" """
    ids = encode(text)[:model.block_size + 1]
    if len(ids) < 2:
        return None, None
    x = torch.tensor([ids[:-1]], device=device)
    y = torch.tensor([ids[1:]], device=device)
    _, loss, _ = model(x, y)
    return loss.item(), math.exp(loss.item())


# ------------------------------------------------------------------ 展示
def show_one(model, prompt, device, args):
    cleaned = clean_prompt(prompt)
    if not cleaned:
        print('  (提示词为空, 从头开始生成)')
    print(f'\n  提示词  : "{cleaned}"')

    if args.show_probs:
        print('  下一个字符的概率分布:')
        for ch, p in next_char_probs(model, cleaned or ' ', device):
            print(f"      '{ch}'  {p * 100:5.1f}%  {'█' * int(p * 40)}")

    for k in range(args.n):
        text = complete(model, cleaned, device, args.words, args.temp, args.top_k)
        tail = text[len(cleaned):].lstrip()
        print(f'  续写 {k + 1:>2} : {cleaned}\033[1m{" " if cleaned else ""}{tail}\033[0m'
              if args.n > 1 else f'  完整句子: {text}')
        if args.score:
            l, ppl = score(model, text, device)
            if l is not None:
                print(f'            (该句平均损失 {l:.3f}, 困惑度 {ppl:.2f})')


def main():
    ap = argparse.ArgumentParser(description='27-token 迷你 Transformer 预测')
    ap.add_argument('--ckpt', default='ckpt.pt', help='train.py 保存的模型参数')
    ap.add_argument('--prompt', type=str, help='提示词; 不给则进入交互模式')
    ap.add_argument('--words', type=int, default=12, help='续写多少个完整单词')
    ap.add_argument('--temp', type=float, default=0.8, help='温度: 越低越保守, 越高越发散')
    ap.add_argument('--top_k', type=int, default=8, help='只在概率最高的 k 个字符里采样')
    ap.add_argument('--n', type=int, default=1, help='一次给出几个候选续写')
    ap.add_argument('--show-probs', action='store_true', help='显示下一个字符的概率分布')
    ap.add_argument('--score', action='store_true', help='给生成结果打困惑度')
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--device', default='auto', help='auto | cpu | cuda | mps')
    args = ap.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    print('=' * 78)
    print('27-Token 迷你 Transformer — 预测')
    print('=' * 78)
    device = pick_device(args.device)

    try:
        model, cfg, stats = load_model(args.ckpt, device)
    except FileNotFoundError:
        print(f'\n找不到模型 {args.ckpt}，请先训练:  python3 train.py')
        return

    print('\n【加载的模型】')
    print(f'  文件           : {args.ckpt}')
    print(f'  结构           : {cfg["n_layer"]} 层 x {cfg["n_head"]} 头, '
          f'd_model={cfg["d_model"]}, 上下文 {cfg["block_size"]}')
    print(f'  参数量         : {model.n_params():,}')
    if stats:
        print(f'  训练情况       : {stats.get("steps", "?")} step / '
              f'{stats.get("seconds", 0):.1f}s @ {stats.get("device", "?")}, '
              f'验证损失 {stats.get("val_loss", float("nan")):.4f}, '
              f'困惑度 {stats.get("ppl", float("nan")):.2f}')
    print(f'\n【采样设置】温度 {args.temp}, top-k {args.top_k}, 续写 {args.words} 个完整单词')

    if args.prompt is not None:
        show_one(model, args.prompt, device, args)
        return

    # 交互模式
    print('\n交互模式：输入提示词回车即可续写；:q 退出，:set 温度 0.5 修改参数')
    while True:
        try:
            line = input('\n提示词> ').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if line in (':q', ':quit', 'exit'):
            break
        if line.startswith(':set'):
            parts = line.split()
            try:
                if '温度' in parts or 'temp' in parts:
                    args.temp = float(parts[-1])
                elif 'words' in parts or '词数' in parts:
                    args.words = int(parts[-1])
                elif 'topk' in parts or 'top_k' in parts:
                    args.top_k = int(parts[-1])
                print(f'  已更新: 温度={args.temp}, 词数={args.words}, top_k={args.top_k}')
            except ValueError:
                print('  用法: :set temp 0.5 | :set words 20 | :set topk 5')
            continue
        show_one(model, line, device, args)


if __name__ == '__main__':
    main()
