"""
数据层：把 26 个英文字母 + 空格 定义成 27 个 token，并提供
    字符 <-> id <-> 矩阵行 的映射，以及训练用的批量采样。

核心思想：
    token id  --one-hot-->  长度 27 的向量  --乘以嵌入矩阵 E(27 x d_model)-->  d_model 维稠密向量
    实际实现中 one-hot @ E 等价于 "取 E 的第 id 行"，即 nn.Embedding 的查表操作。
"""

import random
import torch

# ---------------------------------------------------------------- 词表
# 27 个 token: ' ' + a..z   (空格放在 0 号位, 便于当作 padding / 句子分隔)
ITOS = [' '] + [chr(ord('a') + i) for i in range(26)]
STOI = {ch: i for i, ch in enumerate(ITOS)}
VOCAB_SIZE = len(ITOS)          # 27


def encode(text: str):
    """字符串 -> token id 列表; 非法字符统一折叠成空格"""
    text = text.lower()
    return [STOI.get(ch, 0) for ch in text]


def decode(ids) -> str:
    """token id 列表 -> 字符串"""
    if torch.is_tensor(ids):
        ids = ids.tolist()
    return ''.join(ITOS[i] for i in ids)


def one_hot(ids) -> torch.Tensor:
    """把 id 展开成 one-hot 矩阵 (N x 27), 用于演示 '查表 == one-hot 乘矩阵'"""
    ids = torch.as_tensor(ids, dtype=torch.long)
    return torch.nn.functional.one_hot(ids, num_classes=VOCAB_SIZE).float()


# ---------------------------------------------------------------- 语料
# 用高频英文单词随机拼句子。这样语料里有明确可学的规律：
#   1) 单词的拼写(字母之间的转移)   2) 空格出现的位置(词边界)
# 训练后模型能"拼"出真单词, 学习效果肉眼可见。
WORDS = """the of and to in a is that it for as with was on be at by this have from or
one had not but what all were when we there can an your which their said if do will each
about how up out them then she many some so these would other into has more her two like
him see time could no make than first been its who now people my over know water called
just where most very after thing our name good sentence man think say great help low line
before turn cause same mean differ move right boy old too does tell does set three want air
well also play small end put home read hand port large spell add even land here must big
high such follow act why ask men change went light kind off need house picture try us again
animal point mother world near build self earth father head stand own page should country
found answer school grow study still learn plant cover food sun four between state keep eye
never last let thought city tree cross farm hard start might story saw far sea draw left
late run press close night real life few north open seem together next white children begin
got walk example ease paper often always music those both mark book letter until mile river
car feet care second group carry took rain eat room friend began idea fish mountain stop
once base hear horse cut sure watch color face wood main enough plain girl usual young ready
above ever red list though feel talk bird soon body dog family direct pose leave song measure
door product black short numeral class wind question happen complete ship area half rock order
fire south problem piece told knew pass since top whole king street inch multiply nothing
course stay wheel full force blue object decide surface deep moon island foot system busy
test record boat common gold possible plane stead dry wonder laugh thousand ago ran check game
shape equate hot miss brought heat snow tire bring yes distant fill east paint language among
""".split()


def build_corpus(n_words: int = 40000, seed: int = 1234) -> str:
    """随机拼接单词生成语料; 固定 seed 保证可复现"""
    rng = random.Random(seed)
    return ' '.join(rng.choice(WORDS) for _ in range(n_words))


class CharDataset:
    """把语料切成 (输入 x, 目标 y) 的滑窗样本; y 是 x 右移一位 —— 即 next-token prediction"""

    def __init__(self, text: str, block_size: int, device='cpu', val_ratio: float = 0.1):
        self.block_size = block_size
        self.device = device
        data = torch.tensor(encode(text), dtype=torch.long)
        n = int(len(data) * (1 - val_ratio))
        self.train, self.val = data[:n], data[n:]

    def get_batch(self, split: str, batch_size: int):
        d = self.train if split == 'train' else self.val
        ix = torch.randint(len(d) - self.block_size - 1, (batch_size,))
        x = torch.stack([d[i:i + self.block_size] for i in ix])
        y = torch.stack([d[i + 1:i + 1 + self.block_size] for i in ix])
        return x.to(self.device), y.to(self.device)


if __name__ == '__main__':
    print(f'词表大小 = {VOCAB_SIZE}')
    print('id -> char :', {i: repr(c) for i, c in enumerate(ITOS)})
    s = 'hello world'
    ids = encode(s)
    print(f'\n"{s}" -> {ids} -> "{decode(ids)}"')
    print(f'\none-hot 矩阵形状: {tuple(one_hot(ids).shape)}  (每行 27 维, 只有一个 1)')
    print(one_hot(ids[:3]).int())
