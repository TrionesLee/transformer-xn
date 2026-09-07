# 27-Token 迷你 Transformer

一个**可以完全看懂、可以用计算器手工验算**的 Transformer 语言模型。

把 26 个英文字母 + 空格定义成 27 个 token，映射进一个嵌入矩阵，用从零手写的
decoder-only Transformer（GPT 的最小结构）做下一个字符预测。

## 这个项目解决什么问题

学 Transformer 时最难受的是：论文里的公式看懂了，但对着 `nn.TransformerEncoderLayer`
这一行代码，还是不知道数据到底怎么流过去的、参数到底怎么被改的。本项目的做法是
**把每一层都拆开摊在终端上**：

| 想知道什么 | 这里怎么给你看 |
|---|---|
| token 怎么变成向量 | `walkthrough.py` 打印 one-hot 矩阵与 `E`，并验证 `one_hot @ E == E[ids]` |
| 注意力到底在算什么 | 打印 Q/K/V、原始分数、掩码后的 softmax 权重矩阵 |
| 训练的每一步在干什么 | `--report` 把每步拆成 取数据/前向/反向/更新/验证 5 段 |
| 模型学到了什么 | token 相似度矩阵、注意力热力图、下一个字符的概率分布 |
| **参数是不是黑箱** | **导出 JSON，用纯 Python（不导入 torch）按公式手算，与 PyTorch 对拍到 1e-8** |

不依赖 `nn.TransformerEncoderLayer` / `nn.MultiheadAttention`，注意力、残差、LayerNorm、
FFN、因果掩码全部手写；可视化不依赖 matplotlib，用终端 ASCII 灰度块实现。

## 四个部分

```
  ┌───────────┐   ┌───────────┐   ┌───────────┐   ┌───────────┐
  │ 一、训练   │ → │ 二、预测   │   │ 三、打包   │   │ 四、手算   │
  │ train.py  │   │predict.py │   │ build.sh  │   │handcalc.py│
  └───────────┘   └───────────┘   └───────────┘   └───────────┘
   自动识别设备      提示词续写       免装 Python      纯标准库验算
   ckpt.pt          完整句子         可执行程序       误差 ~1e-8
   params.json
```

## 安装

**只依赖 PyTorch 一个第三方库**，其余全部是 Python 标准库（`argparse` / `json` /
`math` / `os` / `random` / `sys` / `time`）。`handcalc.py` 连 torch 都不需要——
除非加 `--verify` 与 PyTorch 对拍。

```bash
./install.sh              # 建虚拟环境 + 装 CPU 版 torch（约 200MB）
./install.sh --gpu        # 装 CUDA 版 torch（约 2.5GB，需 NVIDIA 显卡）
./install.sh --build      # 额外装 PyInstaller，用于打包可执行程序
./install.sh --here       # 不建虚拟环境，直接装进当前 Python 环境
source .venv/bin/activate # 激活
```

不想用脚本就手动装：

```bash
pip install -r requirements.txt                                              # 自动选版本
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu   # 只要 CPU 版
pip install -r requirements.txt -r requirements-build.txt                    # 连打包依赖一起
```

要求 `torch >= 2.0`（用到的 API 都很基础）。实测 2.7.0+cu128 与 2.14.0+cpu 均正常。

## 60 秒上手

```bash
python3 train.py --input "I like easy course" --report 300   # 训练 + 每步报告
python3 predict.py --prompt "i li"                           # 预测
python3 handcalc.py --prompt "i li" --verify                 # 手算并对拍
```

## 实测能力（默认配置，10 万参数，RTX 4090 上 9 秒）

- 损失 3.2958（= ln27，纯随机）→ **1.17**，困惑度 27 → **3.23**
- `q` 后面预测 `u`：**92.4%**；`peopl` 后面预测 `e`：**100%**
- `wat` 后面 `c`/`e` 各约 50%（watch / water 两条路都对）
- 生成：`bring well song thousand both system wheel make begin most such moon...`

---

# 目录

- [一、这个 Transformer 是怎么工作的](#一这个-transformer-是怎么工作的)
- [二、训练：`train.py`](#二训练trainpy)
- [三、每一步训练在干什么（训练报告）](#三每一步训练在干什么训练报告)
- [四、预测：`predict.py`](#四预测predictpy)
- [五、导出参数并手算概率](#五导出参数并手算概率)
- [六、打包成可执行程序](#六打包成可执行程序)
- [七、文件与方法详解](#七文件与方法详解)  ← **每个文件、每个方法的说明**
- [八、完整实测记录](#八完整实测记录)
- [九、需要留意的几点](#九需要留意的几点)

---

# 一、这个 Transformer 是怎么工作的

## 1.1 为什么是 27 个 token

Transformer 不认识字符，只认识**整数编号**。这里把输入限制成最简单的字母表：

```
' ' -> 0,  'a' -> 1,  'b' -> 2,  ...,  'z' -> 26        共 27 个 token
```

空格排在 0 号，它同时承担"词边界"这个语义——模型要学的第一件事就是"什么时候该
输出空格"。任何输入文本都会被折叠进这 27 个 token：转小写、非 `a-z` 一律变空格、
合并连续空格。

## 1.2 token 如何映射进矩阵

嵌入矩阵 `E` 的形状是 `(27, d_model)`，**每个 token 独占一行**。把 token 变成向量，
本质上是"取出第 id 行"：

```python
one_hot(ids) @ E   ==   E[ids]        # walkthrough.py 里有 allclose 验证
```

左边是 `(T,27)` 的 one-hot 矩阵乘 `(27,D)` 的嵌入矩阵；右边是查表。两者数学上完全
等价，工程上用查表（省掉 99% 的乘零运算）。这就是"把 token 映射到矩阵"的确切含义。

本项目还做了**权重绑定**（`head.weight = tok_emb.weight`）：输出层不再有独立的矩阵，
直接复用 `E`。于是 `E` 的第 i 行有双重身份——既是字符 i 的输入表示，也是"预测字符 i"
的打分向量。最后一步 `logit[i] = E[i] · z` 就是在问：**最终状态 z 和哪个字符的向量最像？**

## 1.3 完整数据流

```
token ids (B,T)                      B=batch, T=序列长, D=d_model, V=27
  │ 查表 E:(27,D)
  ▼
token emb (B,T,D)  +  位置嵌入 P:(T,D)      ← 不加位置，模型分不出 "on" 和 "no"
  ▼
x (B,T,D) ─┬─ LayerNorm → 多头因果自注意力 → 残差相加 ─┐
           └─ LayerNorm → FFN(D→4D→D)      → 残差相加 ─┘  × n_layer
  ▼
LayerNorm → 输出头(即 E 转置) → logits (B,T,27) → 交叉熵(目标 = 输入右移一位)
```

各部件在干什么：

| 部件 | 作用 | 一句话直觉 | 代码位置 |
|---|---|---|---|
| 嵌入 `E` | token → 向量 | 给每个字符一个可学习的坐标 | `MiniTransformer.tok_emb` |
| 位置嵌入 `P` | 位置 → 向量 | 告诉模型"这是第几个字符" | `MiniTransformer.pos_emb` |
| 自注意力 | 位置之间交换信息 | 每个位置去"看"前面哪些字符，按相关度加权取信息 | `MultiHeadSelfAttention` |
| 因果掩码 | 屏蔽未来 | 第 t 个位置只能看 ≤t，否则就是抄答案 | `MultiHeadSelfAttention.mask` |
| FFN | 逐位置非线性变换 | 在每个位置内部做特征加工，参数量的大头 | `FeedForward` |
| 残差 | `x = x + 子层(x)` | 让信息和梯度直通到底层，深层网络才训得动 | `Block.forward` |
| LayerNorm | 归一化 | 稳住每层输入的尺度，防止训练发散 | `Block.ln1/ln2`, `ln_f` |

## 1.4 注意力到底算了什么

对每个位置 t：

```
q[t] = W_q·n[t]     k[t] = W_k·n[t]     v[t] = W_v·n[t]      (n 是 LayerNorm 后的 x)

分数 s[t][j] = (q[t] · k[j]) / sqrt(d_head)        只允许 j ≤ t
权重 a[t][j] = softmax_j(s[t][j])                  每行加起来 = 1
输出 ctx[t]  = Σ_j a[t][j] · v[j]
```

`q` 是"我在找什么"，`k` 是"我能提供什么"，`v` 是"我实际给出的内容"。除以
`sqrt(d_head)` 是为了防止维度变大时点积数值过大、softmax 饱和。

实测（`--tiny` 模型，语料 `i like easy course`，提示词 `"i li"`）：

```
  [头 0] 最后位置对各位置的原始分数:  i:+10.655  ␣:+4.245  l:+12.028  i:+9.872
  [头 0] softmax 后的注意力权重:      i:0.185    ␣:0.000   l:0.730    i:0.085
```

73% 的注意力落在 `l` 上——要预测 `i li` 的下一个字符，关键线索确实是那个 `l`。

## 1.5 训练目标

自监督：目标序列 = 输入序列**右移一位**。

```
输入   "easy course i like easy course i"
目标   "asy course i like easy course i "
对齐   e→a  a→s  s→y  y→␣  ␣→c  c→o  o→u  u→r
```

一条长度 T 的序列一次前向就产生 T 个训练信号（因果掩码保证不会偷看）。损失是
交叉熵，初始值应约等于 `ln(27) = 3.2958`（完全随机猜）。**困惑度** = `exp(loss)`，
可以理解成"模型在 27 个字符里犹豫的等效个数"：3.2958 → 27（全靠蒙），1.17 → 3.2。

---

# 二、训练：`train.py`

## 2.1 自动识别 CPU / GPU

优先级 **CUDA GPU → Apple MPS → CPU**，并打印设备详情；`--device` 可强制指定。

```bash
python3 train.py                     # 自动识别
python3 train.py --device cpu        # 强制 CPU
python3 train.py --device cuda:1     # 指定第 2 块 GPU
```

```
【设备】
  PyTorch 版本   : 2.7.0+cu128
  CUDA 可用      : 是 (共 1 块)
  选用设备       : GPU cuda:0 — NVIDIA GeForce RTX 4090
  显存           : 23.6 GB
  算力 (SM)      : 8.9, 128 个 SM
```

CPU 时会显示逻辑核数与 torch 线程数；无 GPU 时会说明原因（无可用 GPU / torch 未编译
CUDA 支持）。

## 2.2 输入训练文本

四种方式，优先级从上到下：

```bash
python3 train.py --file mytext.txt              # ① 文本文件
python3 train.py --input "hello world ..."      # ② 命令行直接给
python3 train.py --interactive                  # ③ 交互粘贴(连续两个空行结束)
python3 train.py                                # ④ 不给则用内置语料
```

任何文本都会被自动折叠进 27 个 token；文本过短会自动重复以凑足滑窗样本。

## 2.3 全部选项

| 选项 | 默认 | 说明 |
|---|---|---|
| `--file / --input / --interactive` | — | 训练文本来源 |
| `--d_model` | 64 | 嵌入维度，即矩阵 `E` 的列数 |
| `--n_head` | 4 | 注意力头数（须整除 d_model） |
| `--n_layer` | 2 | Transformer 层数 |
| `--block_size` | 32 | 上下文长度（能看多少个字符） |
| `--dropout` | 0.1 | 随机失活比例 |
| `--steps` | 3000 | 训练步数 |
| `--batch_size` | 64 | 每步取多少条序列 |
| `--lr` | 3e-3 | 最大学习率（OneCycle 调度，10% 预热） |
| `--weight_decay` | 0.1 | 权重衰减 |
| `--grad_clip` | 1.0 | 梯度裁剪阈值 |
| `--eval_every` | 250 | 每多少步评估一次 |
| `--eval_iters` | 50 | 每次评估取多少批求平均 |
| `--report` | 0 | **每 N 步打印一份详细训练报告**（0=关闭） |
| `--report-first` | 3 | 开头 K 步无条件出报告 |
| `--export` | params.json | 训练完导出可手算的参数 JSON（空串=不导出） |
| `--tiny` | — | **手算规模预设**：1 层 1 头 d_model=8 上下文 8 dropout=0，共 1144 参数 |
| `--verbose` | — | 额外打印嵌入矩阵/相似度/注意力热力图 |
| `--device` | auto | auto / cpu / cuda / cuda:0 / mps |
| `--out` | ckpt.pt | 模型保存路径 |
| `--seed` | 1337 | 随机种子 |

## 2.4 输出的训练参数

实测（RTX 4090，默认配置，2500 步 / 9.3 秒）：

```
【模型结构】
  层数 x 头数    : 2 x 4   (每头维度 16)
  d_model        : 64        上下文长度: 32        dropout: 0.1
  嵌入矩阵 E     : (27, 64)  <- 27 个 token 各占一行
  总参数量       : 103,488
  各部分参数分布 :
      前馈 FFN          66,176  ( 63.9%)
      注意力            32,896  ( 31.8%)
      位置嵌入 P         2,048  (  2.0%)
      token 嵌入 E       1,728  (  1.7%)
      LayerNorm            640  (  0.6%)
  注: 输出头与 token 嵌入权重绑定, 不额外计参数

  step    train      val      困惑度        lr      耗时   生成样例
     1   3.2795   3.2790    26.55  1.22e-04    0.2s   ' ohhbfszsxfgvvvbcsddnn  ddmnt softgunit'
   500   1.3905   1.3962     4.04  2.91e-03    1.7s   ' cleass force book animalkain starry ago'
  1500   1.1796   1.1862     3.27  1.24e-03    4.6s   ' street main hand help whole measure car'
  2500   1.1664   1.1667     3.21  1.35e-08    9.0s   ' complete press there were start press s'

【训练结果】
  总用时         : 9.3s   (269 step/s, 551k token/s)
  初始损失       : 3.2958 (= ln 27, 完全随机)
  最终验证损失   : 1.1716   困惑度 3.23
  过拟合间隙     : +0.0090  (正常)
  峰值显存       : 41.9 MB

模型参数已保存: ckpt.pt  (421 KB, 103,488 个参数)
参数已导出: params.json  (1517 KB, 29 个张量, 107,264 个数)
```

`ckpt.pt` 供程序加载，`params.json` 供人阅读和手算（见第五节）。

---

# 三、每一步训练在干什么（训练报告）

```bash
python3 train.py --report 500              # 前 3 步 + 之后每 500 步各一份报告
python3 train.py --tiny --report 300       # 手算规模的模型 + 报告
python3 train.py --report 1 --steps 5      # 每一步都出报告
```

每份报告把一步训练拆成 5 个阶段（实现见 [report.py](report.py)）。

### ① 取数据 —— 目标是输入右移一位

```
   batch 形状     : 输入 (64, 32)  目标 (64, 32)  (B=64 条序列 × T=32 个位置)
   本批预测任务数 : 2,048 个 (每个位置都要预测它的下一个字符)
   样本[0] 输入   : "easy course i like easy course i"
   样本[0] 目标   : "asy course i like easy course i "   <- 整体左移了一位
   对齐关系       : e→a  a→s  s→y  y→␣  ␣→c  c→o  o→u  u→r
```

### ② 前向 —— 算出预测和损失

```
   这一批的平均交叉熵 loss = 0.0212   困惑度 = 1.02
   样本[0] 逐位置的预测:
     位置 0 看到 "e"      -> '␣'=0.67  'a'=0.33  'k'=0.00   真实='a'
     位置 1 看到 "ea"     -> 's'=1.00  '␣'=0.00  'i'=0.00   真实='s' ✓
     位置 2 看到 "eas"    -> 'y'=1.00  'e'=0.00  'a'=0.00   真实='y' ✓
```

位置 0 那个 0.67/0.33 不是没学好——语料 `i like easy course` 里三个 `e`
（lik**e**、**e**asy、cours**e**）恰好 2 个后接空格、1 个后接 `a`，模型学到的正是
**真实条件分布**。

### ③ 反向 —— 每个参数拿到 ∂loss/∂w

```
   梯度总范数     : 0.0379  (未超过阈值 1.0，不裁剪)
   梯度最大的 4 个参数张量:
     tok_emb.weight                   |grad| = 0.01981
     blocks.0.attn.qkv.weight         |grad| = 0.01586
   嵌入矩阵 E 里梯度最大的 token 行:
     '␣'=0.012*   'i'=0.010*   'e'=0.009*   'c'=0.005*   'k'=0.004*
     (带 * 表示该 token 出现在本批数据里)
```

梯度最大的是本批出现过的字符——**这就是"学习"发生的位置**。梯度总范数超过
`--grad_clip` 时会等比缩放，报告里会明确写出是否触发。

### ④ 更新 —— AdamW 按梯度改参数

```
   参数总变化量   : |Δw| = 0.07281   占参数总规模 |w| = 22.870 的 0.318%
   变化最大的 3 个参数张量:
     blocks.1.attn.qkv.weight         |Δw| = 0.03143
   嵌入矩阵 E 改动最大的 token 行:
     'c'=0.0023*   'k'=0.0020*   '␣'=0.0020*   'a'=0.0016*
```

一步只改动千分之几——学习是几千次微调累积出来的。

### ⑤ 验证 —— 同一批数据再前向一次

```
   更新前 loss = 0.0212   更新后 loss = 0.0194   ↓ 下降 0.0018
```

闭环确认这一步确实起了作用。

> 想看**单次前向/反向的每个中间张量**（one-hot、Q/K/V、注意力矩阵、GELU 输出……），
> 用 `python3 walkthrough.py`，它把一次完整的前向+反向拆成 7 步逐个打印。

---

# 四、预测：`predict.py`

```bash
python3 predict.py --prompt "the qu"            # 单次续写
python3 predict.py                              # 交互模式，反复输入提示词
python3 predict.py --prompt "peo" --words 20    # 续写 20 个完整单词
python3 predict.py --prompt "the" --n 5         # 一次给 5 个候选
python3 predict.py --prompt "wat" --show-probs  # 附下一个字符的概率分布
python3 predict.py --prompt "the" --temp 0.3    # 低温度 = 更保守
python3 predict.py --prompt "the" --score       # 给生成结果打困惑度
python3 predict.py --ckpt demo_ckpt.pt --prompt "i li"   # 指定模型文件
```

生成**按完整单词收尾**，不会截在半个单词处。实测：

```
  提示词  : "the qu"
  下一个字符的概率分布:
      'e'   99.7%  ███████████████████████████████████████
      'a'    0.2%
  完整句子: the question complete car cause could can we those them black had cut
            (该句平均损失 0.968, 困惑度 2.63)
```

| 选项 | 默认 | 说明 |
|---|---|---|
| `--ckpt` | ckpt.pt | 模型文件 |
| `--prompt` | — | 提示词；不给则进入交互模式 |
| `--words` | 12 | 续写多少个完整单词 |
| `--temp` | 0.8 | 温度：越低越保守，越高越发散 |
| `--top_k` | 8 | 只在概率最高的 k 个字符里采样 |
| `--n` | 1 | 一次给几个候选续写 |
| `--show-probs` | — | 显示下一个字符的概率分布 |
| `--score` | — | 给生成结果打困惑度 |
| `--seed` | — | 固定随机种子，让采样可复现 |
| `--device` | auto | auto / cpu / cuda / mps |

交互模式下可随时改参数：`:set temp 0.5`、`:set words 20`、`:set topk 5`，`:q` 退出。

---

# 五、导出参数并手算概率

这是本项目的重点：**训练出来的参数不是黑箱，拿计算器就能复现它的输出。**

## 5.1 导出

训练结束会自动导出 `params.json`；也可以单独导出：

```bash
python3 export_params.py --ckpt ckpt.pt --out params.json
python3 export_params.py --ckpt tiny_ckpt.pt --out tiny.json --nd 12 --print
```

JSON 里有四块：

| 字段 | 内容 |
|---|---|
| `meta` | 27 个 token 的映射表、结构配置（d_model / n_head / d_head / eps / 权重绑定）、训练信息 |
| `formula` | 手算要用的 12 条公式，照着套即可 |
| `tensor_desc` | 每个矩阵的形状和用途说明 |
| `tensors` | 全部参数数值（默认保留 6 位小数，`--nd` 可调） |

## 5.2 手算公式（导出文件里的 `formula` 字段）

```
1. 分词      : 字符 c -> token id  (' '=0, 'a'=1, ..., 'z'=26)
2. 嵌入      : x[t] = E[id_t] + P[t]
3. LayerNorm : n = (x - mean(x)) / sqrt(var(x) + 1e-5) * γ + β     # var 除以 D（有偏）
4. QKV       : q = W_q·n , k = W_k·n , v = W_v·n                   # qkv.weight 的三段
5. 注意力分数: s[t][j] = (q[t]·k[j]) / sqrt(d_head)  , 仅 j ≤ t
6. softmax   : a[t][j] = exp(s[t][j] - max) / Σ_j exp(s[t][j] - max)
7. 加权求和  : ctx[t] = Σ_j a[t][j] * v[j]
8. 输出投影  : x = x + (W_O·ctx + b_O)
9. FFN       : h = GELU(W1·LN2(x) + b1) ; x = x + (W2·h + b2)
               GELU(u) = 0.5 * u * (1 + erf(u / sqrt(2)))
10. 最终归一 : z = LayerNorm_f(x[最后一个位置])
11. 打分     : logit[i] = E[i] · z          # 权重绑定，输出头就是嵌入矩阵
12. 概率     : P(下一个字符 = i) = exp(logit[i] - max) / Σ_k exp(logit[k] - max)
```

## 5.3 用手算规模的模型

默认模型有 10 万参数，手算不现实。用 `--tiny` 训一个 **1144 参数**的版本
（1 层 1 头 d_model=8 上下文 8，dropout 关闭）：

```bash
python3 train.py --tiny --input "I like easy course" --steps 1500 \
                 --out demo_tiny.pt --export demo_tiny.json
```

导出的 JSON 只有 25 KB、17 个张量、1424 个数——完全可以逐个核对。

## 5.4 逐步演算

`handcalc.py` **只用 Python 标准库**（`math` + `json`），不导入 PyTorch：

```bash
python3 handcalc.py --params demo_tiny.json --prompt "i li"                     # 完整推导
python3 handcalc.py --params demo_tiny.json --prompt "i li" --brief             # 只看结果
python3 handcalc.py --params demo_tiny.json --ckpt demo_tiny.pt --verify        # 与 PyTorch 对拍
python3 handcalc.py --params demo_tiny.json --prompt "i li" --ncol 8 --topk 5   # 显示更多分量
```

真实输出（节选）：

```
【第 1 步】分词: "i li"  ->  [9, 0, 12, 9]

【第 2 步】嵌入: x[t] = E[id_t] + P[t]
  位置3 'i':  E[ 9] = [-0.6341, -0.2893, -0.5467, +0.0015, +0.8978, -0.4666, ...]
            + P[ 3] = [-0.0638, +0.0997, +0.0734, +0.0081, -0.0917, +0.0027, ...]
            = x[ 3]  = [-0.6979, -0.1895, -0.4733, +0.0096, +0.8061, -0.4638, ...]

【第 3 步】LayerNorm
    均值 = -0.094607   方差 = 0.249860   1/sqrt(方差+eps) = 2.000520

【第 5~7 步】注意力: 分数 = q·k / sqrt(8) = q·k * 0.353553
  [头 0] 原始分数:  i:+10.655  ␣:+4.245  l:+12.028  i:+9.872
  [头 0] 注意力权重: i:0.185    ␣:0.000   l:0.730    i:0.085

【第 11 步】打分: logit['k'] = E[11] · z = +8.799452

【第 12 步】softmax
    最大 logit = 8.799452，分母 Σexp(logit - 最大值) = 1.008101
    P('k') = exp(+8.7995 - 8.7995) / 1.0081 = 0.991964 = 99.20%
  27 个概率之和 = 1.000000  (应为 1)
```

## 5.5 对拍结果

`--verify` 把手算结果和 PyTorch 逐个比对：

```
    字符           手算      PyTorch            差
     k   0.99196363   0.99196362     9.62e-09
     i   0.00311654   0.00311655     1.62e-08
27 个概率的最大绝对误差 = 3.152e-08
结论: 一致（误差仅来自导出时的小数截断）
```

三个规模都对得上：手算模型 **3.15e-08**、10 万参数模型 **3.63e-08**、
打包版训练的模型 **6.34e-08**。**误差只来自 JSON 导出时保留 6 位小数**，
`--nd 12` 可进一步缩小。

---

# 六、打包成可执行程序

> **仓库里没有现成的可执行程序。** PyInstaller 产物体积过大（CUDA 版 6.4 GB，
> 单文件版也有 258 MB），已在 `.gitignore` 中排除；而且可执行文件绑定平台——
> Linux 上打的包在 Windows 用不了。需要时用下面的命令现场生成即可，
> 依赖装好后一条命令几分钟就出来。

```bash
./install.sh --build            # 先装打包依赖（PyInstaller）
./build.sh                      # 目录版(推荐): dist/minitf/minitf，带 CUDA，可用 GPU
./build.sh --cpu                # 精简版: 用 CPU-only 的 torch，体积小 8 倍
./build.sh --onefile --cpu      # 单文件版: dist/minitf-onefile，拷走即用
./build.sh --name mytool        # 自定义可执行文件名
```

打好后用子命令操作，**目标机器不需要装 Python 或 PyTorch**：

```bash
dist/minitf/minitf                                  # 查看全部子命令
dist/minitf/minitf info                             # 环境、设备、已有模型信息
dist/minitf/minitf vocab                            # 27 个 token 的映射表
dist/minitf/minitf train --file mytext.txt --report 500
dist/minitf/minitf predict --prompt "the qu"
dist/minitf/minitf export --ckpt ckpt.pt --out params.json
dist/minitf/minitf handcalc --prompt "the c" --verify
dist/minitf/minitf walkthrough --text "the cat"
dist/minitf/minitf inspect --prefix "the qu"
```

实测体积与行为：

| 构建方式 | 体积 | 设备 | 启动 |
|---|---|---|---|
| `./build.sh` | 6.4 GB（目录） | 自动识别，实测 `GPU cuda:0 — RTX 4090` | ~10 s |
| `./build.sh --cpu` | 854 MB（目录） | `CPU — 24 逻辑核` | ~2 s |
| `./build.sh --onefile --cpu` | **258 MB（单个文件）** | CPU | ~5 s |

体积主要来自 PyTorch：CUDA 版光是 GPU 动态库就有 5 GB 以上。

**为什么 `--cpu` 不是简单删库**：CUDA 版 torch 在 `import` 阶段就硬依赖
`libtorch_cuda.so`，从包里删掉它会直接 `ImportError`。所以 `--cpu` 走另一条路——自动
建一个 `.venv-cpu` 虚拟环境装 CPU-only 的 torch 轮子再打包（首次需联网下载约 200 MB，
不污染主 Python 环境）。

**Windows 下生成 .exe**：装好 Python + PyTorch + PyInstaller 后执行

```powershell
pyinstaller --clean --noconfirm minitf.spec
```

得到 `dist\minitf\minitf.exe`，子命令用法完全一致。

**打包环境的两处临时处理**（`build.sh` 自动完成，结束时**一律自动还原**）：

1. `pathlib` 向后兼容包 —— Python 2 时代遗留，PyInstaller 明确拒绝在其存在时打包；
2. conda 的 `backports/__init__.py` —— 它把 `backports` 变成普通包，遮蔽了 setuptools
   依赖的 `backports.tarfile`，会让 PyInstaller 的 setuptools 钩子崩溃。

---

# 七、文件与方法详解

## 7.0 模块依赖关系

```
                    ┌──────────┐
                    │ data.py  │  词表 / 编码 / 语料 / 批采样   ← 无依赖，最底层
                    └────┬─────┘
                         │
                    ┌────▼─────┐
                    │ model.py │  MiniTransformer 网络结构
                    └────┬─────┘
         ┌───────────────┼────────────────┬─────────────────┐
    ┌────▼─────┐   ┌─────▼──────┐   ┌─────▼──────┐   ┌──────▼───────┐
    │ train.py │   │ predict.py │   │walkthrough │   │inspect_model │
    └────┬─────┘   └─────┬──────┘   └────────────┘   └──────┬───────┘
         │               │ 复用 train.pick_device           │
    ┌────▼─────┐    ┌────▼──────────┐                       │
    │report.py │    │export_params  │───► params.json ──┐   │
    └──────────┘    └───────────────┘                   │   │
    ┌──────────┐                                   ┌────▼───▼───┐
    │visualize │◄── train / inspect_model 调用      │ handcalc.py│ 纯标准库
    └──────────┘                                   └────────────┘ 不依赖 torch

                    ┌──────────┐
                    │  cli.py  │  统一子命令入口，分发到上面所有模块
                    └──────────┘
                         ▲
                 build.sh + minitf.spec  打包成可执行程序
```

---

## `data.py` — 数据层（98 行）

把 26 个字母 + 空格定义成 27 个 token，提供 字符 ↔ id ↔ 矩阵行 的映射和批采样。
**无任何项目内依赖，是最底层模块。**

### 常量

| 名称 | 内容 |
|---|---|
| `ITOS` | id → 字符的列表：`[' ', 'a', 'b', ..., 'z']`，长度 27 |
| `STOI` | 字符 → id 的字典，`ITOS` 的反向表 |
| `VOCAB_SIZE` | `27`，词表大小 |
| `WORDS` | 356 个高频英文单词，供 `build_corpus()` 拼语料 |

### 方法

**`encode(text) -> list[int]`**
字符串转 token id 列表。先转小写，不在词表里的字符统一折叠成空格（id 0）。
`encode("Hello!")` → `[8, 5, 12, 12, 15, 0]`。

**`decode(ids) -> str`**
token id 列表（或张量）转回字符串，`encode` 的逆操作。接受 list 或 `torch.Tensor`。

**`one_hot(ids) -> Tensor(N, 27)`**
把 id 展开成 one-hot 矩阵，每行只有一个 1。**只用于演示"查表 == one-hot 乘矩阵"**
这个等价关系，训练时不走这条路（那样会浪费 99% 的乘零运算）。

**`build_corpus(n_words=40000, seed=1234) -> str`**
从 `WORDS` 里随机抽词拼成语料，用空格连接。固定 seed 保证可复现。默认 4 万个词，
生成 213,520 个字符。这个语料**有拼写规律但没有语法**——模型能学会拼单词和放空格，
学不出句法。

### 类 `CharDataset`

把语料切成滑窗样本，供训练循环取批。

**`__init__(text, block_size, device='cpu', val_ratio=0.1)`**
把文本编码成一维 token 张量，按 90%/10% 切成 `self.train` / `self.val`。
`block_size` 是每条样本的长度（上下文窗口）。

**`get_batch(split, batch_size) -> (x, y)`**
随机取 `batch_size` 个起点，切出 `(batch_size, block_size)` 的输入 `x` 和目标 `y`。
**`y` 就是 `x` 右移一位**——这是 next-token prediction 的全部秘密。返回的张量已放到
`device` 上。

### 直接运行

```bash
python3 data.py     # 打印词表、编码示例、one-hot 矩阵
```

---

## `model.py` — 网络结构（180 行）

从零实现 decoder-only Transformer。**刻意不用 `nn.TransformerEncoderLayer`**，
把每步矩阵运算展开。

### 类 `MultiHeadSelfAttention`

多头因果自注意力：`Attention(Q,K,V) = softmax(QKᵀ/√d_k + mask)·V`

**`__init__(d_model, n_head, block_size, dropout=0.1)`**
- `self.qkv`：一个 `Linear(D, 3D)`，一次矩阵乘法同时算出 Q、K、V（比三个独立
  Linear 快）
- `self.proj`：`Linear(D, D)`，多头拼接后的输出投影
- `self.mask`：注册为 buffer 的下三角矩阵 `(1,1,block_size,block_size)`，
  **因果掩码**，保证位置 t 只能看到 ≤t

**`forward(x, return_attn=False) -> (y, att_w)`**
- 输入 `(B,T,D)`，输出同形状
- 拆头：`(B,T,D)` → `(B, n_head, T, d_head)`
- 分数除以 `√d_head`，上三角填 `-inf` 后 softmax
- `return_attn=True` 时同时返回注意力权重 `(B, n_head, T, T)` 供可视化

### 类 `FeedForward`

**`__init__(d_model, dropout=0.1)`** / **`forward(x)`**
逐位置前馈网络 `D → 4D → GELU → D`，在每个位置内部做非线性特征加工。
**参数量的大头**（默认配置占 63.9%）。

### 类 `Block`

一个 Transformer 层 = 注意力子层 + 前馈子层，都是 **Pre-LN + 残差**。

**`forward(x, return_attn=False) -> (x, att)`**
```python
x = x + attn(ln1(x))     # 残差①
x = x + ffn(ln2(x))      # 残差②
```
Pre-LN（归一化放在子层之前）比 Post-LN 更容易训练，不需要学习率预热也能稳定。

### 类 `MiniTransformer`

**`__init__(d_model=64, n_head=4, n_layer=2, block_size=32, dropout=0.1)`**
- `tok_emb`：**嵌入矩阵 E `(27, D)`** —— 就是"27 个 token 映射到的矩阵"
- `pos_emb`：可学习位置嵌入 `(block_size, D)`
- `blocks`：`n_layer` 个 `Block`
- `ln_f`：输出前最后一次 LayerNorm
- `head`：`Linear(D, 27, bias=False)`，**权重与 `tok_emb` 绑定**

**`_init_weights(m)`**（静态方法）
Linear 和 Embedding 用 `N(0, 0.02)` 初始化，bias 置零。初始化对不对，看初始 loss
是否≈`ln(27)=3.2958` 就知道。

**`forward(idx, targets=None, return_attn=False) -> (logits, loss, attns)`**
- `idx (B,T)` → `logits (B,T,27)`
- 给了 `targets` 就顺带算交叉熵（把 `(B,T,27)` 摊平成 `(B*T,27)`）
- `return_attn=True` 时 `attns` 是每层的注意力权重列表

**`generate(idx, max_new_tokens, temperature=1.0, top_k=None) -> Tensor`**
自回归采样。每次只保留最近 `block_size` 个字符做条件，取最后一个位置的 logits，
温度缩放 → top-k 过滤 → softmax → `multinomial` 采样 → 拼回输入。
带 `@torch.no_grad()`，内部自动切换 eval/train 模式。

**`n_params() -> int`**
参数总量（权重绑定的输出头不重复计数）。

### 直接运行

```bash
python3 model.py    # 打印参数量、各张量形状、验证初始 loss ≈ ln(27)
```

---

## `train.py` — 第一部分：训练（320 行）

### `pick_device(want='auto') -> torch.device`

**自动识别计算设备**并打印详情。优先级 CUDA → MPS(Apple) → CPU；
`want` 不是 `'auto'` 时直接用指定的。GPU 会打印型号、显存、SM 数量；CPU 会打印
逻辑核数、torch 线程数，以及"为什么没用 GPU"（无可用 GPU / torch 未编译 CUDA）。
**`predict.py` 和 `inspect_model.py` 都复用这个函数**，保证三处行为一致。

### `load_text(args) -> (clean_text, source_desc)`

按优先级取训练文本：`--file` > `--input` > `--interactive` > 内置语料。然后统一清洗：
转小写 → 非 `a-z` 变空格 → 合并连续空格。**文本短于 200 字符时自动重复**
（重复到约 4000 字符），否则滑窗样本不够。返回清洗后的文本和来源描述。

### `parse_args() -> Namespace`

定义全部命令行选项，分成四组：文本输入、模型结构、训练超参、训练报告与导出、其它。
完整选项表见 [2.3 全部选项](#23-全部选项)。

### `estimate_loss(model, ds, batch_size, iters) -> {'train':…, 'val':…}`

在 eval 模式下各取 `iters` 批求平均损失，比单批的瞬时 loss 稳定得多。
带 `@torch.no_grad()`，算完自动切回 train 模式。

### `main()`

完整训练流程，八个阶段：

1. `--tiny` 时覆盖结构参数（`d_model=8, n_head=1, n_layer=1, block_size=8, dropout=0`）
2. `pick_device()` 识别设备
3. `load_text()` + `CharDataset` 准备数据，打印语料统计（含"实际出现了哪些 token"）
4. 建模型，**打印参数分布表**（按 FFN / 注意力 / 位置嵌入 / token 嵌入 / LayerNorm 分组）
5. 打印训练超参 + 训练前的随机生成（对照组）
6. **训练循环**：`get_batch` → `forward` → `backward` → `clip_grad_norm_` → `opt.step()`
   → `sched.step()`，其中：
   - 用 `OneCycleLR` 调度（10% 预热后余弦衰减）
   - 每步维护 EMA 平滑损失供画曲线
   - `--report` 命中的步调用 `report.step_report()`
   - 每 `--eval_every` 步评估一次并生成一段样例
7. 打印损失曲线、训练结果统计（速度、过拟合间隙、峰值显存）
8. 保存 `ckpt.pt`（含 `args` / `config` / `model` / `stats` 四块）+ 调用
   `export_json()` 导出 `params.json`

---

## `report.py` — 每步训练报告（129 行）

被 `train.py` 在 `--report` 命中的步调用。

### `_c(i) -> str`
token id 转可打印字符，空格显示成 `␣`（否则在报告里看不见）。

### `capture_params(model) -> dict[str, Tensor]`
**更新前**的参数快照（`.detach().clone()`）。必须在 `opt.step()` 之前调用，
之后用来算这一步参数被改了多少。

### `step_report(step, model, x, y, loss_before, total_norm, clipped, lr, before, sample_row=0, topk=3)`

打印一步训练的完整报告，**调用时机是 `opt.step()` 之后**。五个阶段：

| 参数 | 含义 |
|---|---|
| `step` | 第几步 |
| `x, y` | 本步的输入与目标 `(B,T)` |
| `loss_before` | 更新前的前向损失 |
| `total_norm` | `clip_grad_norm_` 返回的裁剪**前**梯度总范数 |
| `clipped` | 裁剪阈值（`--grad_clip`），用来判断是否触发 |
| `lr` | 本步实际使用的学习率 |
| `before` | `capture_params()` 的快照 |
| `sample_row` | 展示 batch 里的第几条样本 |
| `topk` | 每个位置显示前几名预测 |

内部会**再前向一次**（用更新后的模型跑同一批），得到 ⑤ 段的"更新后 loss"，
形成闭环验证。报告内容详见 [第三节](#三每一步训练在干什么训练报告)。

---

## `predict.py` — 第二部分：预测（190 行）

### `load_model(path, device) -> (model, cfg, stats)`
读 ckpt，优先用 `config` 字段重建模型（旧文件回退到 `args`），返回模型、结构配置和
训练统计（步数、耗时、验证损失、困惑度）。

### `clean_prompt(s) -> str`
提示词也要折叠进 27 个 token：转小写、非 `a-z` 变空格、合并连续空格。

### `complete(model, prompt, device, words=12, temperature=0.8, top_k=8, max_chars=400) -> str`

**核心生成函数**。从提示词开始自回归采样，每一步：

```
取最近 block_size 个字符 → 前向 → 取最后位置的 logits
→ 温度缩放 → top-k 过滤 → softmax → multinomial 采样 → 拼回
```

**遇到空格就计一个完整单词**，凑满 `words` 个才停——所以输出永远不会截在半个单词处。
`max_chars` 是保险上限，防止模型不吐空格时死循环。

### `next_char_probs(model, prompt, device, topk=8) -> list[(char, prob)]`
只做一次前向，返回下一个字符概率最高的 topk 个。`--show-probs` 用它。

### `score(model, text, device) -> (loss, perplexity)`
给一段文本打分：平均每字符交叉熵和困惑度，衡量模型觉得它有多"像英语"。
`--score` 用它评估自己生成的句子。

### `show_one(model, prompt, device, args)`
把上面几个拼起来输出：提示词 → （可选）概率分布 → `--n` 个候选续写 →（可选）困惑度。

### `main()`
解析参数 → `pick_device()` → `load_model()` → 打印模型信息与采样设置 →
有 `--prompt` 就单次输出，否则进**交互模式**（支持 `:set temp/words/topk`、`:q`）。

---

## `export_params.py` — 参数导出（135 行）

### 常量

- **`DESC`**：每个参数张量的用途说明字典（"这是 Q/K/V 三段拼一起的投影矩阵"这类）
- **`FORMULA`**：手算用的 12 条公式，原样写进导出的 JSON

### `_round(t, nd=6)`
张量递归转成嵌套 list 并保留 `nd` 位小数——JSON 不能直接存张量，而且全精度浮点
读起来很累。

### `export_json(model, path='params.json', nd=6, extra=None) -> dict`

把模型全部参数写成 JSON，四个顶层字段：`meta`（词表 + 结构配置 + 训练信息）、
`formula`（12 条公式）、`tensor_desc`（每个张量的形状和用途）、`tensors`（数值）。
返回 `{kb, n_tensors, n_scalars, config}` 供调用方打印。
`extra` 用来塞训练信息（步数、验证损失、设备、语料字符数）。

### `main()`
独立命令行入口：从 `--ckpt` 读模型 → 导出 → 打印结构摘要和 12 条公式。
`--print` 会把全部参数数值打到屏幕（只适合 `--tiny` 规模）。

---

## `handcalc.py` — 手算验算（286 行）★

**本项目最有说服力的部分：只用 `math` 和 `json`，一行 torch 都不导入**，
按公式把概率算出来，证明导出的参数就是全部信息。

### 基础线性代数（都是几行的纯 Python）

| 方法 | 作用 |
|---|---|
| `dot(a, b)` | 向量点积 |
| `matvec(M, v)` | 矩阵（行优先）乘向量，`结果[i] = M[i]·v` |
| `vadd(a, b)` | 向量逐元素相加 |
| `layer_norm(x, gamma, beta, eps)` | 返回 `(归一化结果, 均值, 方差)`。**方差用有偏估计（除以 D）**，和 PyTorch 一致 |
| `softmax(xs)` | 减最大值防溢出的标准实现 |
| `gelu(u)` | `0.5u(1+erf(u/√2))`，**精确版**，和 `nn.GELU` 默认行为一致（不是 tanh 近似） |

### `forward(P, ids, trace=None) -> list[float]`

纯 Python 跑完整前向，返回最后一个位置对 27 个 token 的概率。

- `P` 是 `json.load()` 出来的参数字典
- `trace` 传字典进去时，把所有中间结果（嵌入、LayerNorm 统计、Q/K/V、
  注意力权重、ctx、FFN 中间值、logits）塞进去供打印
- 逐层遍历 `blocks.{i}.*`，多头就按 `[h*d_head : (h+1)*d_head]` 切片分别算
- **因果掩码的实现是 `range(t+1)`**——只对 ≤t 的位置算分数，比乘掩码矩阵更直白
- 最后 `logit[i] = dot(E[i], z)`，利用权重绑定，输出头就是嵌入矩阵

### `fmt(v, n=6, nd=4) -> str`
向量格式化，只显示前 n 个分量（`d_model=64` 时全打出来没法读）。

### `show(P, prompt, ids, trace, probs, ncol=6, topk=8)`
把 `trace` 里的中间结果按 12 个步骤打印成推导过程，每步都带公式说明。

### `main()`
读 JSON → 清洗提示词 → `forward()` → `show()` 或 `--brief` 简报 →
`--verify` 时加载 PyTorch 模型对拍，逐个字符列出手算值、PyTorch 值和差值。

---

## `walkthrough.py` — 单次前向+反向逐步走查（159 行）

### `sep(title)` / `brief(t, n=6)`
分隔线打印；张量取前 n 个元素格式化。

### `main()`
把**一次**前向传播 + 一次参数更新拆成 7 步打印：

| 步骤 | 打印内容 |
|---|---|
| 第 0 步 | 分词：字符 → token id → 还原验证 |
| 第 1 步 | 查表嵌入，**验证 `one_hot @ E` 与 `E[ids]` 用 `allclose` 相等** |
| 第 2 步 | 加位置编码，展示同一字母在不同位置得到不同向量 |
| 第 3 步 | Q/K/V 形状、**原始注意力分数矩阵**、掩码后 softmax 的权重矩阵（带字符标注） |
| 第 4 步 | 逐层输出的均值/标准差，说明残差的作用 |
| 第 5 步 | logits → softmax，逐位置列出 top-3 预测与真实答案 |
| 第 6 步 | 交叉熵、困惑度、**每个参数张量的梯度范数**、嵌入矩阵每行的梯度排序、更新一步后的 loss 变化 |

默认用随机初始化的超小模型（`d_model=16, 1 层 2 头`）让数字看得清；
`--ckpt ckpt.pt` 可以换成训练好的模型，注意力就有意义了。

---

## `inspect_model.py` — 模型内部探查（74 行）

### `load(path='ckpt.pt', device=None) -> (model, cfg, device)`
加载 ckpt 并重建模型。`device=None` 时调用 `train.pick_device('auto')` 自动识别。
**`handcalc.py --verify` 和 `walkthrough.py --ckpt` 都复用这个函数。**

### `main()`
免重训地探查已训练模型：

1. 演示 `E[ids]` 与 `one_hot(ids) @ E` 相等，并打印 "cat" 三个字符的向量
2. 嵌入矩阵热力图
3. token 余弦相似度矩阵 + 每个 token 的最近邻
4. 各层注意力热力图（`--layer` / `--head` 指定）
5. 指定前缀的下一个字符概率分布
6. 自由生成一段文本

---

## `visualize.py` — 终端可视化（113 行）

**不依赖 matplotlib**（本机 matplotlib 因 numpy 2.x ABI 冲突已损坏），用 10 级灰度块
字符 `SHADES = ' .:-=+*#%@'` 画热力图。

| 方法 | 作用 |
|---|---|
| `_label(i)` | token id → 显示标签，空格显示成 `_` |
| `heatmap(mat, row_labels, col_labels, title, vmin, vmax, cell, robust)` | 通用 2D 热力图。`robust=True` 时按 `mean±2std` 裁剪值域，避免少数极值把对比度吃光 |
| `show_embedding(model, title)` | 嵌入矩阵 `E` 的热力图，27 行各标字符 |
| `show_similarity(model, topk=4)` | token 余弦相似度矩阵 + 每个 token 的最近邻列表 |
| `show_attention(model, ids, device, layer, head)` | 某层某头的注意力矩阵，行=查询位置，列=被关注位置，上三角因掩码恒为 0 |
| `show_next_char_dist(model, prefix, device, topk=8)` | 给定前缀，下一个字符的概率条形图 |
| `loss_curve(history, width=64, height=12)` | ASCII 折线图，把损失序列重采样到指定宽度 |

---

## `cli.py` — 统一子命令入口（141 行）

打包成可执行程序后就是 `minitf`。**不含业务逻辑，只做分发**：改写 `sys.argv`
后调用对应模块的 `main()`，这样每个子模块仍能单独运行。

| 子命令 | 分发到 | 别名 |
|---|---|---|
| `train` | `train.main()` | |
| `predict` | `predict.main()` | `gen`, `generate` |
| `walkthrough` | `walkthrough.main()` | `walk`, `steps` |
| `inspect` | `inspect_model.main()` | `show` |
| `export` | `export_params.main()` | |
| `handcalc` | `handcalc.main()` | `calc` |
| `info` | `cmd_info()` | |
| `vocab` | `cmd_vocab()` | |

### `cmd_info(argv)`
打印 Python 版本、**运行方式（脚本 / 打包的可执行程序）**、工作目录，调用
`pick_device()` 显示设备，再读 ckpt 打印模型结构与训练统计。

### `cmd_vocab(argv)`
打印 27 个 token 的完整映射表，以及给定字符串的编码结果和 one-hot 形状。

### `main()`
取 `sys.argv[1]` 作为子命令，把剩余参数交给子模块的 argparse。无参数或 `-h`
时打印带示例的用法说明。

---

## `build.sh` — 打包脚本

| 选项 | 作用 |
|---|---|
| （无） | 目录版，用当前环境的 torch（有 CUDA 就带 CUDA） |
| `--onefile` | 打成单个可执行文件 |
| `--cpu` | 用 CPU-only 的 torch（自动建 `.venv-cpu` 并安装） |
| `--name X` | 自定义可执行文件名 |

### `restore_env()`（trap EXIT/INT/TERM）
打包前把两个会让 PyInstaller 崩溃的文件临时移到 `mktemp -d` 目录，
**脚本结束时（含出错、Ctrl-C）自动还原**：

1. `site-packages/pathlib.py` + `pathlib-*.dist-info` —— Python 2 时代的向后兼容包
2. `site-packages/backports/__init__.py` —— 遮蔽了 setuptools 依赖的 `backports.tarfile`
   （只在检测到 `import setuptools` 确实失败时才移走）

## `minitf.spec` — PyInstaller 配置

通过环境变量控制（由 `build.sh` 设置）：`MINITF_ONEFILE` / `MINITF_CPU` / `MINITF_NAME`。
用 `collect_all('torch')` 收集 PyTorch，`EXCLUDES` 排除本项目用不到的大件
（matplotlib、scipy、nltk、pyarrow 等）——**不排除的话，环境里任何一个装坏了的包
都会中断打包**。`MINITF_CPU=1` 时还会过滤掉 CUDA 相关的二进制。

---

## `requirements.txt` / `requirements-build.txt` / `install.sh` — 依赖与安装

- **`requirements.txt`**：运行依赖，只有 `torch>=2.0` 和 `numpy>=1.24`
  （numpy 项目本身不用，只是为了消掉 torch 缺它时每次运行都打印的警告）。
  文件头注释里写了 CPU / CUDA 两种装法。
- **`requirements-build.txt`**：打包才需要的 `pyinstaller>=6.0`，用 `-r` 继承运行依赖。
- **`install.sh`**：一键建 venv 并安装。`--gpu` 装 CUDA 版，`--build` 连打包依赖一起装，
  `--here` 不建 venv 直接装进当前环境，`--venv DIR` 自定义目录。
  装完会实测打印 Python 版本、PyTorch 版本和 CUDA 是否可用。

## 生成的文件

以下都**不入库**，需要时用对应命令重新生成：

| 文件 | 内容 | 重新生成 |
|---|---|---|
| `ckpt.pt` | PyTorch 权重文件，含 `args` / `config` / `model` / `stats` 四块 | `python3 train.py` |
| `params.json` | 人类可读的参数导出，含词表、公式、张量说明和全部数值 | `python3 export_params.py` |
| `dist/`、`build/` | 打包出的可执行程序（GB 级） | `./build.sh` |
| `.venv/`、`.venv-cpu/` | 虚拟环境 | `./install.sh` / `./build.sh --cpu` |

例外：`tiny_params.json` 和 `demo_tiny.json`（各约 25 KB）**入库保留**，
它们是 README 手算章节的示例数据，体积小且免得读者非要先训练才能看懂。

---

# 八、完整实测记录

以 `I like easy course` 为训练文本跑通全流程（2026-09-07，RTX 4090）。

### 1. 训练

```bash
python3 train.py --input "I like easy course" --steps 1500 --report 700 \
                 --out demo_ckpt.pt --export demo_params.json
```

文本清洗：`I`→`i`，18 字符自动重复 222 次得到 4218 字符，只用到 12 个 token
`_aceiklorsuy`。

```
     1   3.2589   3.2590    26.02      ' ohhbfszqqllfffoczwj x  ddmst softgenitnntnno'
   300   0.0219   0.0214     1.02      ' i like easy course i like easy course i like'
  1500   0.0188   0.0195     1.02      ' i like easy course i like easy course i like'
```

**300 步完全学会**：困惑度 26.02 → 1.02（1.0 是理论下限），耗时 6.0 秒。

### 2. 预测

```
提示词 "i li"    -> 'k' 100.0%   完整句子: i like easy course i like easy course i
提示词 "e"       -> '␣' 65.1%  'a' 34.8%
提示词 "easy c"  -> 完整句子: easy course i like easy course i
```

`e` 那个 65/35 正是语料的真实分布（3 个 e 里 2 个后接空格）。

### 3. 手算对拍

```bash
python3 train.py --tiny --input "I like easy course" --steps 1500 \
                 --out demo_tiny.pt --export demo_tiny.json     # 1144 参数
python3 handcalc.py --params demo_tiny.json --ckpt demo_tiny.pt --prompt "i li" --verify
```

```
    字符           手算      PyTorch            差
     k   0.99196363   0.99196362     9.62e-09
27 个概率的最大绝对误差 = 3.152e-08
```

### 4. 打包版

`dist/minitf`（GPU）跑通 训练→预测→导出→手算对拍 全链路；
`dist/minitf-onefile`（258 MB 单文件，CPU）输出 `i like easy course i like`。

---

# 九、需要留意的几点

- **内置语料没有语法**。它是把高频英文单词随机拼接而成的，所以模型只能学会"拼对单词"
  和"在词边界放空格"，学不出句法。想看语法就用 `--file` 换成真实文本。
- **`--tiny` 模型是为手算准备的，不是为质量准备的**。1144 个参数在内置语料上只能训到
  验证损失 2.26（困惑度 9.56）；默认模型是 1.17（困惑度 3.23）。手算演示用 tiny，
  看效果用默认配置。
- **`handcalc.py` 只复现推理，不含反向传播**。想看梯度怎么算，看 `walkthrough.py`
  的第 6 步和训练报告的第 ③ 段。
- **验证损失与训练损失基本重合**，说明这个规模下没有过拟合。用内置语料时把 `--steps`
  调大只会在 1.16 附近收敛——这是语料熵的下限，不是没训好。
- **手算的误差来自导出精度**，不是算法差异。默认保留 6 位小数导致 ~1e-8 的偏差，
  `--nd 12` 可以进一步缩小。
- **matplotlib 在本机不可用**（scipy/matplotlib 是针对 numpy 1.x 编译的，而 numpy 是
  2.4.6），所以全部可视化都用终端 ASCII 实现，不依赖任何绘图库。
