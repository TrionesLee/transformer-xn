"""
第一部分：训练
    - 自动识别 CPU / GPU（也可用 --device 强制指定）
    - 输入要训练的文本（文件 / 命令行字符串 / 交互粘贴 / 内置语料）
    - 输出训练好的模型参数到 ckpt.pt，并打印全部训练参数与统计

用法:
    python3 train.py                              # 用内置语料
    python3 train.py --file mytext.txt            # 用文本文件训练
    python3 train.py --input "hello world ..."    # 直接给一段文本
    python3 train.py --interactive                # 交互粘贴文本(空行两次结束)
    python3 train.py --device cpu --steps 1000    # 强制 CPU
    python3 train.py --verbose                    # 额外打印嵌入矩阵/注意力等可视化
"""

import argparse
import math
import os
import sys
import time
import torch

from data import CharDataset, build_corpus, encode, decode, ITOS, VOCAB_SIZE
from model import MiniTransformer
import visualize as viz
import report
from export_params import export_json


# ------------------------------------------------------------------ 设备识别
def pick_device(want='auto'):
    """自动识别可用的计算设备并打印详细信息"""
    if want != 'auto':
        dev = torch.device(want)
    elif torch.cuda.is_available():
        dev = torch.device('cuda')
    elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        dev = torch.device('mps')                     # Apple Silicon
    else:
        dev = torch.device('cpu')

    print('【设备】')
    print(f'  PyTorch 版本   : {torch.__version__}')
    if dev.type == 'cuda':
        i = dev.index or 0
        p = torch.cuda.get_device_properties(i)
        print(f'  CUDA 可用      : 是 (共 {torch.cuda.device_count()} 块)')
        print(f'  选用设备       : GPU cuda:{i} — {p.name}')
        print(f'  显存           : {p.total_memory / 1024**3:.1f} GB')
        print(f'  算力 (SM)      : {p.major}.{p.minor}, {p.multi_processor_count} 个 SM')
    elif dev.type == 'mps':
        print('  选用设备       : Apple MPS (GPU)')
    else:
        cores = os.cpu_count()
        cuda_why = '未编译 CUDA 支持' if not torch.backends.cuda.is_built() else '无可用 GPU'
        print(f'  CUDA 可用      : 否 ({cuda_why})')
        print(f'  选用设备       : CPU — {cores} 逻辑核, torch 线程数 {torch.get_num_threads()}')
    return dev


# ------------------------------------------------------------------ 文本输入
def load_text(args):
    """按优先级获取训练文本: --file > --input > --interactive > 内置语料"""
    if args.file:
        with open(args.file, encoding='utf-8', errors='ignore') as f:
            raw = f.read()
        src = f'文件 {args.file}'
    elif args.input:
        raw = args.input
        src = '命令行 --input'
    elif args.interactive:
        print('请粘贴训练文本，输入完成后连续两个空行结束（Ctrl-D 也可）:')
        lines, blanks = [], 0
        for line in sys.stdin:
            if line.strip() == '':
                blanks += 1
                if blanks >= 2:
                    break
            else:
                blanks = 0
            lines.append(line)
        raw = ''.join(lines)
        src = '交互输入'
    else:
        raw = build_corpus()
        src = '内置语料(高频英文单词随机拼接)'

    # 折叠成 27 个 token 能表示的形式: 小写, 非 a-z 一律变空格, 合并连续空格
    clean = ''.join(c if 'a' <= c <= 'z' else ' ' for c in raw.lower())
    clean = ' '.join(clean.split())
    if len(clean) < 200:
        # 文本太短不足以训练, 重复拼接以保证有足够的滑窗样本
        rep = max(2, 4000 // max(1, len(clean)))
        clean = (clean + ' ') * rep
        print(f'  提示: 文本过短, 已重复 {rep} 次以凑足训练样本')
    return clean, src


def parse_args():
    p = argparse.ArgumentParser(description='27-token 迷你 Transformer 训练')
    g = p.add_argument_group('文本输入')
    g.add_argument('--file', type=str, help='训练文本文件路径')
    g.add_argument('--input', type=str, help='直接给一段训练文本')
    g.add_argument('--interactive', action='store_true', help='交互粘贴训练文本')

    g = p.add_argument_group('模型结构')
    g.add_argument('--d_model', type=int, default=64, help='嵌入维度(矩阵 E 的列数)')
    g.add_argument('--n_head', type=int, default=4, help='注意力头数')
    g.add_argument('--n_layer', type=int, default=2, help='Transformer 层数')
    g.add_argument('--block_size', type=int, default=32, help='上下文长度')
    g.add_argument('--dropout', type=float, default=0.1)

    g = p.add_argument_group('训练超参')
    g.add_argument('--steps', type=int, default=3000)
    g.add_argument('--batch_size', type=int, default=64)
    g.add_argument('--lr', type=float, default=3e-3)
    g.add_argument('--weight_decay', type=float, default=0.1)
    g.add_argument('--grad_clip', type=float, default=1.0)
    g.add_argument('--eval_every', type=int, default=250)
    g.add_argument('--eval_iters', type=int, default=50)
    g.add_argument('--seed', type=int, default=1337)

    g = p.add_argument_group('训练报告与导出')
    g.add_argument('--report', type=int, default=0, metavar='N',
                   help='每 N 步打印一份详细的训练步骤报告(0=关闭)')
    g.add_argument('--report-first', type=int, default=3, metavar='K',
                   help='开头 K 步无条件打印报告(配合 --report 使用)')
    g.add_argument('--export', type=str, default='params.json', metavar='FILE',
                   help='把训练完的全部参数导出为可手算的 JSON(空字符串=不导出)')
    g.add_argument('--tiny', action='store_true',
                   help='手算规模预设: 1层1头 d_model=8 上下文8, 共 1144 个参数')

    g = p.add_argument_group('其它')
    g.add_argument('--device', default='auto', help='auto | cpu | cuda | cuda:0 | mps')
    g.add_argument('--out', default='ckpt.pt', help='模型参数保存路径')
    g.add_argument('--verbose', action='store_true', help='额外打印嵌入矩阵/注意力可视化')
    return p.parse_args()


@torch.no_grad()
def estimate_loss(model, ds, batch_size, iters):
    model.eval()
    res = {}
    for split in ('train', 'val'):
        losses = torch.zeros(iters)
        for k in range(iters):
            x, y = ds.get_batch(split, batch_size)
            _, loss, _ = model(x, y)
            losses[k] = loss.item()
        res[split] = losses.mean().item()
    model.train()
    return res


def main():
    args = parse_args()
    if args.tiny:                    # 手算规模: 参数少到可以用计算器逐步验算
        args.d_model, args.n_head, args.n_layer = 8, 1, 1
        args.block_size, args.dropout = 8, 0.0
    torch.manual_seed(args.seed)
    print('=' * 78)
    print('27-Token 迷你 Transformer — 训练')
    print('=' * 78)

    device = pick_device(args.device)

    # ---------------- 文本 ----------------
    text, src = load_text(args)
    ds = CharDataset(text, args.block_size, device=device)
    used = sorted(set(encode(text)))
    print('\n【训练文本】')
    print(f'  来源           : {src}')
    print(f'  清洗后字符数   : {len(text):,}   词数 {len(text.split()):,}')
    print(f'  词表           : {VOCAB_SIZE} 个 token = 空格 + a..z')
    print(f'  实际出现       : {len(used)} 个 token — '
          f'{"".join("_" if ITOS[i] == " " else ITOS[i] for i in used)}')
    print(f'  训练 / 验证    : {len(ds.train):,} / {len(ds.val):,} tokens')
    print(f'  开头预览       : "{text[:70]}..."')

    # ---------------- 模型 ----------------
    model = MiniTransformer(args.d_model, args.n_head, args.n_layer,
                            args.block_size, args.dropout).to(device)
    print('\n【模型结构】')
    print(f'  层数 x 头数    : {args.n_layer} x {args.n_head}   '
          f'(每头维度 {args.d_model // args.n_head})')
    print(f'  d_model        : {args.d_model}')
    print(f'  上下文长度     : {args.block_size}')
    print(f'  dropout        : {args.dropout}')
    print(f'  嵌入矩阵 E     : ({VOCAB_SIZE}, {args.d_model})  <- 27 个 token 各占一行')
    print(f'  总参数量       : {model.n_params():,}')
    print('  各部分参数分布 :')
    groups = {}
    for n, p in model.named_parameters():
        key = ('token 嵌入 E' if 'tok_emb' in n else
               '位置嵌入 P' if 'pos_emb' in n else
               '注意力' if 'attn' in n else
               '前馈 FFN' if 'ffn' in n else
               'LayerNorm')
        groups[key] = groups.get(key, 0) + p.numel()
    for k, v in sorted(groups.items(), key=lambda kv: -kv[1]):
        print(f'      {k:<14} {v:>9,}  ({v / model.n_params() * 100:5.1f}%)')
    print('  注: 输出头与 token 嵌入权重绑定, 不额外计参数')

    # ---------------- 训练超参 ----------------
    print('\n【训练超参】')
    print(f'  优化器         : AdamW (betas=0.9/0.99, weight_decay={args.weight_decay})')
    print(f'  学习率         : {args.lr:g}  (OneCycle 调度, 10% 预热)')
    print(f'  batch / steps  : {args.batch_size} / {args.steps}')
    print(f'  梯度裁剪       : {args.grad_clip}')
    print(f'  随机种子       : {args.seed}')

    if args.verbose:
        print('\n【训练前的嵌入矩阵】')
        print(viz.show_embedding(model, '嵌入矩阵 E (训练前: 随机噪声, 无结构)'))

    x0 = torch.zeros((1, 1), dtype=torch.long, device=device)
    print('\n  训练前生成:', repr(decode(model.generate(x0, 100, top_k=10)[0])))

    # ---------------- 训练循环 ----------------
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(0.9, 0.99), weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                                total_steps=args.steps, pct_start=0.1)
    curve, ema, best = [], None, float('inf')
    t0 = time.time()
    print('\n【训练】前向 -> 交叉熵 -> 反向传播 -> AdamW 更新参数')
    print('-' * 78)
    print(f'{"step":>6} {"train":>8} {"val":>8} {"困惑度":>8} {"lr":>9} {"耗时":>7}   生成样例')
    print('-' * 78)

    if args.report:
        print(f'  (已开启训练步骤报告: 前 {args.report_first} 步 + 之后每 {args.report} 步各一份)')

    for step in range(1, args.steps + 1):
        # 这一步要不要出详细报告
        rep = bool(args.report) and (step <= args.report_first or step % args.report == 0)
        snapshot = report.capture_params(model) if rep else None

        x, y = ds.get_batch('train', args.batch_size)              # ① 取数据
        _, loss, _ = model(x, y)                                   # ② 前向
        opt.zero_grad(set_to_none=True)
        loss.backward()                                            # ③ 反向
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        lr_now = sched.get_last_lr()[0]
        opt.step()                                                 # ④ 更新
        sched.step()

        if rep:                                                    # ⑤ 报告
            report.step_report(step, model, x, y, loss.item(), gnorm.item(),
                               args.grad_clip, lr_now, snapshot)

        ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
        curve.append((step, ema))

        if step % args.eval_every == 0 or step == 1:
            l = estimate_loss(model, ds, args.batch_size, args.eval_iters)
            best = min(best, l['val'])
            sample = decode(model.generate(x0, 44, top_k=10)[0])
            print(f'{step:>6} {l["train"]:>8.4f} {l["val"]:>8.4f} '
                  f'{math.exp(l["val"]):>8.2f} {sched.get_last_lr()[0]:>9.2e} '
                  f'{time.time() - t0:>6.1f}s   {sample!r}')

    dt = time.time() - t0
    final = estimate_loss(model, ds, args.batch_size, args.eval_iters)
    print('-' * 78)
    print(viz.loss_curve(curve))

    # ---------------- 训练结果 ----------------
    print('\n【训练结果】')
    print(f'  总用时         : {dt:.1f}s   ({args.steps / dt:.0f} step/s, '
          f'{args.steps * args.batch_size * args.block_size / dt / 1e3:.0f}k token/s)')
    print(f'  初始损失       : {math.log(VOCAB_SIZE):.4f} (= ln 27, 完全随机)')
    print(f'  最终训练损失   : {final["train"]:.4f}')
    print(f'  最终验证损失   : {final["val"]:.4f}   困惑度 {math.exp(final["val"]):.2f}')
    print(f'  最好验证损失   : {best:.4f}')
    gap = final['val'] - final['train']
    print(f'  过拟合间隙     : {gap:+.4f}  '
          f'({"正常" if gap < 0.05 else "验证损失偏高, 可加大 dropout 或减小模型"})')
    if device.type == 'cuda':
        print(f'  峰值显存       : {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB')

    if args.verbose:
        print(viz.show_embedding(model, '嵌入矩阵 E (训练后)'))
        print(viz.show_similarity(model))
        probe = encode('the quick brown')[:args.block_size]
        print(viz.show_attention(model, probe, device, layer=0, head=0))

    print('\n  训练后生成    :', repr(decode(model.generate(x0, 100, top_k=8)[0])))

    # ---------------- 保存 ----------------
    torch.save({
        'args': vars(args),
        'config': dict(d_model=args.d_model, n_head=args.n_head, n_layer=args.n_layer,
                       block_size=args.block_size, dropout=args.dropout),
        'model': model.state_dict(),
        'stats': dict(steps=args.steps, seconds=dt, train_loss=final['train'],
                      val_loss=final['val'], ppl=math.exp(final['val']),
                      n_params=model.n_params(), chars=len(text), device=str(device)),
    }, args.out)
    size = os.path.getsize(args.out) / 1024
    print(f'\n模型参数已保存: {args.out}  ({size:.0f} KB, {model.n_params():,} 个参数)')

    # 导出成人类可读、可手算的 JSON
    if args.export:
        info = export_json(model, args.export,
                           extra=dict(steps=args.steps, val_loss=final['val'],
                                      device=str(device), chars=len(text)))
        print(f'参数已导出: {args.export}  ({info["kb"]:.0f} KB, '
              f'{info["n_tensors"]} 个张量, {info["n_scalars"]:,} 个数)')
        print(f'  用它手算下一个字符的概率:  python3 handcalc.py --prompt "the c"')
    # 打包成可执行程序后, 提示对应的子命令写法
    if getattr(sys, 'frozen', False):
        hint = f'{os.path.basename(sys.executable)} predict --prompt "the qu"'
    else:
        hint = 'python3 predict.py --prompt "the qu"'
    print(f'下一步用它做预测:  {hint}')


if __name__ == '__main__':
    main()
