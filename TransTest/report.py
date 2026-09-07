"""
每一步训练的详细报告：把"这一步到底干了什么"拆成 5 个阶段打印出来。

    ① 取数据   从语料随机截 batch，构造 (输入, 目标) —— 目标是输入右移一位
    ② 前向     嵌入 → 注意力 → FFN → logits → 交叉熵，看具体某个位置预测得对不对
    ③ 反向     每个参数拿到 ∂loss/∂w，看谁的梯度大、是否触发梯度裁剪
    ④ 更新     AdamW 按梯度改参数，看哪些 token 的嵌入行被改得最多
    ⑤ 验证     用同一批数据重新前向，确认 loss 确实下降了
"""

import math
import torch

from data import ITOS, decode


def _c(i):
    """token id -> 可打印字符（空格显示成 ␣）"""
    return '␣' if ITOS[i] == ' ' else ITOS[i]


def capture_params(model):
    """更新前的参数快照，用于计算这一步参数被改了多少"""
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def step_report(step, model, x, y, loss_before, total_norm, clipped,
                lr, before, sample_row=0, topk=3):
    """
    打印一步训练的完整报告。调用时机：optimizer.step() 之后。

    step         第几步
    x, y         本步的输入与目标 (B, T)
    loss_before  更新前的前向损失
    total_norm   裁剪前的梯度总范数
    clipped      梯度裁剪的阈值
    lr           本步实际用的学习率
    before       更新前的参数快照 (capture_params 的返回值)
    """
    B, T = x.shape
    xs, ys = x[sample_row].tolist(), y[sample_row].tolist()
    W = 78
    print('\n' + '─' * W)
    print(f'第 {step} 步训练报告'.center(W - 4))
    print('─' * W)

    # ---------------- ① 取数据 ----------------
    print('① 取数据 —— 从语料里随机截取 batch，目标 = 输入右移一位')
    print(f'   batch 形状     : 输入 {tuple(x.shape)}  目标 {tuple(y.shape)}  '
          f'(B={B} 条序列 × T={T} 个位置)')
    print(f'   本批预测任务数 : {B * T:,} 个 (每个位置都要预测它的下一个字符)')
    print(f'   样本[{sample_row}] 输入   : "{decode(xs)}"')
    print(f'   样本[{sample_row}] 目标   : "{decode(ys)}"   <- 整体左移了一位')
    print(f'   对齐关系       : ' + '  '.join(
        f'{_c(a)}→{_c(b)}' for a, b in list(zip(xs, ys))[:8]) + '  ...')

    # ---------------- ② 前向 ----------------
    print('\n② 前向传播 —— 27 词表查表嵌入 → 自注意力 → 前馈网络 → logits → 交叉熵')
    with torch.no_grad():
        model.eval()
        logits, _, _ = model(x[sample_row:sample_row + 1])
        model.train()
        probs = torch.softmax(logits[0], dim=-1)
    print(f'   这一批的平均交叉熵 loss = {loss_before:.4f}   '
          f'困惑度 = {math.exp(loss_before):.2f}')
    print(f'   (困惑度 = 模型在 27 个字符里"犹豫"的等效个数，纯随机时为 27)')
    print(f'   样本[{sample_row}] 逐位置的预测（更新后的模型）:')
    show = min(6, T)
    for t in range(show):
        p, idx = probs[t].topk(topk)
        pred = '  '.join(f"'{_c(j)}'={v:.2f}" for v, j in zip(p.tolist(), idx.tolist()))
        hit = '✓' if idx[0].item() == ys[t] else ' '
        print(f'     位置{t:>2} 看到 "{decode(xs[:t+1])}" -> {pred}   '
              f"真实='{_c(ys[t])}' {hit}")
    if T > show:
        print(f'     ... 其余 {T - show} 个位置省略')

    # ---------------- ③ 反向 ----------------
    print('\n③ 反向传播 —— 链式法则算出每个参数的梯度 ∂loss/∂w')
    grads = [(n, p.grad.norm().item()) for n, p in model.named_parameters()
             if p.grad is not None]
    grads.sort(key=lambda kv: -kv[1])
    print(f'   梯度总范数     : {total_norm:.4f}'
          + (f'  -> 超过阈值 {clipped}，已等比缩放（防止梯度爆炸）'
             if total_norm > clipped else f'  (未超过阈值 {clipped}，不裁剪)'))
    print('   梯度最大的 4 个参数张量:')
    for n, g in grads[:4]:
        print(f'     {n:<32} |grad| = {g:.5f}')
    ge = model.tok_emb.weight.grad.norm(dim=1)
    in_batch = set(x.flatten().tolist())
    top_tok = ge.argsort(descending=True)[:6].tolist()
    print('   嵌入矩阵 E 里梯度最大的 token 行:')
    print('     ' + '   '.join(
        f"'{_c(i)}'={ge[i]:.3f}{'*' if i in in_batch else ''}" for i in top_tok))
    print('     (带 * 表示该 token 出现在本批数据里)')

    # ---------------- ④ 更新 ----------------
    print(f'\n④ 参数更新 —— AdamW 按梯度方向改参数，本步学习率 lr = {lr:.3e}')
    tot_delta, tot_w = 0.0, 0.0
    rows = []
    for n, p in model.named_parameters():
        d = (p.detach() - before[n]).norm().item()
        tot_delta += d ** 2
        tot_w += p.detach().norm().item() ** 2
        rows.append((n, d))
    rows.sort(key=lambda kv: -kv[1])
    print(f'   参数总变化量   : |Δw| = {math.sqrt(tot_delta):.5f}   '
          f'占参数总规模 |w| = {math.sqrt(tot_w):.3f} 的 '
          f'{math.sqrt(tot_delta) / math.sqrt(tot_w) * 100:.3f}%')
    print('   变化最大的 3 个参数张量:')
    for n, d in rows[:3]:
        print(f'     {n:<32} |Δw| = {d:.5f}')
    dE = (model.tok_emb.weight.detach() - before['tok_emb.weight']).norm(dim=1)
    top_row = dE.argsort(descending=True)[:6].tolist()
    print('   嵌入矩阵 E 改动最大的 token 行:')
    print('     ' + '   '.join(
        f"'{_c(i)}'={dE[i]:.4f}{'*' if i in in_batch else ''}" for i in top_row))

    # ---------------- ⑤ 验证 ----------------
    with torch.no_grad():
        model.eval()
        _, loss_after, _ = model(x, y)
        model.train()
    d = loss_before - loss_after.item()
    print(f'\n⑤ 效果验证 —— 用同一批数据再前向一次')
    print(f'   更新前 loss = {loss_before:.4f}   更新后 loss = {loss_after.item():.4f}   '
          f'{"↓ 下降" if d > 0 else "↑ 上升"} {abs(d):.4f}')
    print(f'   把这个"取数据→前向→反向→更新"的循环重复几千次，就是完整的训练过程。')
    print('─' * W)
