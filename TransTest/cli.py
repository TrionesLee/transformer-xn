#!/usr/bin/env python3
"""
统一命令行入口 —— 打包成可执行程序后, 用子命令操作:

    minitf train      [选项]    训练模型 (自动识别 CPU/GPU, 输入文本, 输出参数)
    minitf predict    [选项]    用训练好的模型, 输入提示词续写完整句子
    minitf walkthrough[选项]    逐步走查一次前向 + 反向传播的全部中间结果
    minitf inspect    [选项]    探查模型内部: 嵌入矩阵 / 注意力 / 预测分布
    minitf info                 显示环境、设备与已有模型信息
    minitf vocab                打印 27 个 token 的映射表

每个子命令的详细选项:  minitf <子命令> --help
"""

import os
import sys

PROG = os.path.basename(sys.argv[0]) or 'minitf'

USAGE = f"""27-Token 迷你 Transformer  —  训练 / 预测 一体化工具

用法: {PROG} <子命令> [选项]

子命令:
  train         训练模型。自动识别 CPU/GPU，可从文件/命令行/交互输入训练文本，
                训练完成后保存 ckpt.pt 并导出可手算的 params.json
                  {PROG} train
                  {PROG} train --file mytext.txt --steps 5000
                  {PROG} train --input "hello world" --device cpu
                  {PROG} train --report 500          # 每 500 步出一份详细训练报告
                  {PROG} train --tiny --report 300   # 手算规模(1144 参数)+ 步骤报告

  predict       加载模型，输入提示词，自回归续写出完整句子
                  {PROG} predict --prompt "the qu"
                  {PROG} predict --prompt "peo" --words 20 --temp 0.6
                  {PROG} predict                    # 交互模式，反复输入提示词

  walkthrough   逐步走查：把一次前向传播 + 一次参数更新的每个中间张量打印出来
                  {PROG} walkthrough --text "the cat"
                  {PROG} walkthrough --ckpt ckpt.pt --text "the quick"

  inspect       探查训练好的模型：嵌入矩阵热力图、token 相似度、注意力权重
                  {PROG} inspect --prefix "the quick brown"

  export        把训练好的参数导出成可读、可手算的 JSON
                  {PROG} export --ckpt ckpt.pt --out params.json

  handcalc      只用标准库(不加载 PyTorch)按公式手算下一个字符的概率，
                每一步中间数值都打印出来，可与 PyTorch 对拍
                  {PROG} handcalc --prompt "the c"
                  {PROG} handcalc --prompt "the c" --verify

  info          显示 PyTorch 版本、可用设备、已有模型的训练信息
  vocab         打印 27 个 token（空格 + a..z）的映射表

查看某个子命令的全部选项:  {PROG} <子命令> --help
"""


def cmd_info(argv):
    import torch
    from train import pick_device
    print('=' * 78)
    print('27-Token 迷你 Transformer — 环境信息')
    print('=' * 78)
    print(f'  Python         : {sys.version.split()[0]}')
    print(f'  运行方式       : ' +
          ('打包的可执行程序' if getattr(sys, 'frozen', False) else 'Python 脚本'))
    print(f'  工作目录       : {os.getcwd()}\n')
    pick_device('auto')

    ck = argv[0] if argv else 'ckpt.pt'
    print(f'\n【模型文件 {ck}】')
    if not os.path.exists(ck):
        print(f'  不存在。先训练一个:  {PROG} train')
        return 0
    d = torch.load(ck, map_location='cpu', weights_only=False)
    c, s = d.get('config', {}), d.get('stats', {})
    print(f'  大小           : {os.path.getsize(ck) / 1024:.0f} KB')
    print(f'  结构           : {c.get("n_layer")} 层 x {c.get("n_head")} 头, '
          f'd_model={c.get("d_model")}, 上下文 {c.get("block_size")}')
    print(f'  参数量         : {s.get("n_params", 0):,}')
    print(f'  训练           : {s.get("steps")} step / {s.get("seconds", 0):.1f}s '
          f'@ {s.get("device")}, 语料 {s.get("chars", 0):,} 字符')
    print(f'  验证损失       : {s.get("val_loss", float("nan")):.4f}  '
          f'困惑度 {s.get("ppl", float("nan")):.2f}')
    return 0


def cmd_vocab(argv):
    from data import ITOS, STOI, VOCAB_SIZE, encode, one_hot
    print(f'词表大小 = {VOCAB_SIZE} （空格 + 26 个字母）\n')
    for i, ch in enumerate(ITOS):
        name = '空格' if ch == ' ' else ch
        print(f'  id {i:>2}  ->  {name}', end='\n' if (i + 1) % 4 == 0 else '     ')
    print('\n\n嵌入过程: token id -> one-hot(27维) -> 乘嵌入矩阵 E(27 x d_model) -> 稠密向量')
    s = ' '.join(argv) if argv else 'hello world'
    print(f'\n  "{s}"  ->  {encode(s)}')
    print(f'  one-hot 矩阵形状: {tuple(one_hot(encode(s)).shape)}')
    return 0


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help', 'help'):
        print(USAGE)
        return 0

    cmd, rest = sys.argv[1], sys.argv[2:]
    # 让子命令自己的 argparse 看到干净的 argv
    sys.argv = [f'{PROG} {cmd}'] + rest

    if cmd == 'train':
        import train
        return train.main()
    if cmd in ('predict', 'gen', 'generate'):
        import predict
        return predict.main()
    if cmd in ('walkthrough', 'walk', 'steps'):
        import walkthrough
        return walkthrough.main()
    if cmd in ('inspect', 'show'):
        import inspect_model
        return inspect_model.main()
    if cmd == 'export':
        import export_params
        return export_params.main()
    if cmd in ('handcalc', 'calc'):
        import handcalc
        return handcalc.main()
    if cmd == 'info':
        return cmd_info(rest)
    if cmd == 'vocab':
        return cmd_vocab(rest)

    print(f'未知子命令: {cmd}\n')
    print(USAGE)
    return 2


if __name__ == '__main__':
    sys.exit(main() or 0)
