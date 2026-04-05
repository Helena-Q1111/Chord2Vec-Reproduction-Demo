# Chord2Vec - 和弦嵌入学习

基于 [chord2vec](https://github.com/Sephora-M/chord2vec) 论文的 PyTorch 现代实现，使用 Hooktheory 数据集训练和弦向量表示。

## 概述

Chord2Vec 使用类似 word2vec 的 skip-gram 架构学习和弦的向量表示。通过预测相邻和弦，模型学习到和弦之间的功能和声关系。

### 核心思想

- **Skip-gram 模型**: 给定一个和弦，预测其上下文和弦
- **负采样**: 使用负采样进行高效训练
- **和弦嵌入**: 学习到的向量捕获和弦之间的音乐关系
  - 相似功能的和弦（如 I 和 VI）会有相近的向量
  - 可以进行和弦类比运算（如：I:V :: IV:?）

## 项目结构

```
chord2vec/
├── __init__.py          # 包初始化
├── model.py             # 模型定义 (SkipGram, Linear, Seq2Seq)
├── data_processing.py   # 数据预处理
├── train.py             # 训练脚本
├── visualize.py         # 可视化与评估
├── requirements.txt     # 依赖项
└── README.md            # 本文档
```

## 安装

```bash
# 安装依赖
cd chord2vec
pip install -r requirements.txt
```

## 快速开始

### 1. 训练模型

```bash
# 基本训练（使用 Hooktheory 数据集）
python train.py --data ../Hooktheory.json.gz --epochs 50

# 完整参数
python train.py \
    --data ../Hooktheory.json.gz \
    --output_dir output \
    --model_type skipgram \
    --embedding_dim 128 \
    --window_size 2 \
    --epochs 50 \
    --batch_size 256 \
    --lr 0.001 \
    --n_negative 5
```

### 2. 可视化与评估

```bash
# 运行所有分析
python visualize.py --model_dir output --all

# 单独运行
python visualize.py --model_dir output --tsne        # t-SNE 可视化
python visualize.py --model_dir output --similarity  # 相似性分析
python visualize.py --model_dir output --cluster     # 和弦聚类
python visualize.py --model_dir output --history     # 训练曲线
```

### 3. Python API 使用

```python
import torch
import numpy as np
from chord2vec import SkipGramChord2Vec, load_processed_data

# 加载训练好的模型
checkpoint = torch.load('output/best_model.pt')
model = SkipGramChord2Vec(
    vocab_size=checkpoint['vocab_size'],
    embedding_dim=checkpoint['embedding_dim']
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 加载词汇表
import json
with open('output/vocabulary.json', 'r') as f:
    vocab = json.load(f)
chord2idx = vocab['chord2idx']
idx2chord = {int(k): v for k, v in vocab['idx2chord'].items()}

# 查找相似和弦
similar = model.most_similar(chord2idx['V'], idx2chord, top_k=5)
print("与 V 最相似的和弦:")
for chord, score in similar:
    print(f"  {chord}: {score:.3f}")

# 和弦类比
from chord2vec import compute_analogy
# I:V :: IV:? (预期: I 或类似的主和弦)
results = compute_analogy(model, idx2chord, chord2idx, 'I', 'V', 'IV', top_k=5)
print("\nI:V :: IV:?")
for chord, score in results:
    print(f"  {chord}: {score:.3f}")

# 获取和弦嵌入
embeddings = model.get_all_embeddings()
print(f"\n嵌入矩阵形状: {embeddings.shape}")
```

### 4. 纯前端 demo

`demo/` 目录下提供了一个静态可听 demo：

- 打开 `demo/index.html`
- 以 `output_ring_v4` 的 embeddings 和词汇表为数据源
- 点击和弦可播放钢琴音色，第二次点击会显示 cosine 相似度和连线

## 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data` | Hooktheory.json.gz | 输入数据文件 |
| `--output_dir` | output | 输出目录 |
| `--model_type` | skipgram | 模型类型 (skipgram/linear) |
| `--embedding_dim` | 128 | 嵌入维度 |
| `--window_size` | 2 | 上下文窗口大小 |
| `--epochs` | 50 | 训练轮数 |
| `--batch_size` | 256 | 批次大小 |
| `--lr` | 0.001 | 学习率 |
| `--n_negative` | 5 | 负采样数量 |
| `--min_count` | 2 | 最小和弦出现次数 |
| `--early_stopping` | 5 | 早停耐心值 |

## 输出文件

训练完成后，`output/` 目录包含：

- `best_model.pt` - 最佳模型检查点
- `chord_embeddings.npy` - 和弦嵌入矩阵 (NumPy 格式)
- `vocabulary.json` - 和弦词汇表（chord2idx/idx2chord 映射）
- `training_history.json` - 训练历史（损失曲线等）
- `processed_data.pkl` - 预处理后的数据（可复用）

## 模型架构

### 1. SkipGramChord2Vec (默认)

标准 skip-gram 模型，使用负采样：
- 输入：中心和弦
- 输出：预测上下文和弦
- 损失：负采样对比损失

### 2. LinearChord2Vec

线性模型，假设和弦中各音符条件独立：
- 输入：和弦的 one-hot 编码
- 输出：上下文和弦的概率分布
- 损失：二元交叉熵

### 3. Seq2SeqChord2Vec

基于 LSTM 的编码器-解码器模型：
- 编码器：将输入和弦序列编码为固定向量
- 解码器：从向量生成上下文和弦序列
- 适用于更长的上下文建模

## 数据格式

Hooktheory 数据集使用音阶度数表示和弦：
- `I` - 主和弦 (Tonic)
- `II` - 上主和弦 (Supertonic)
- `III` - 中音和弦 (Mediant)
- `IV` - 下属和弦 (Subdominant)
- `V` - 属和弦 (Dominant)
- `VI` - 下中音和弦 (Submediant)
- `VII` - 导和弦 (Leading tone)

和弦可能包含额外信息如 `7`（七和弦）、`m`（小调）等。

## 参考

- 原始 Chord2Vec 实现: https://github.com/Sephora-M/chord2vec
- Word2Vec 论文: Mikolov et al., "Efficient Estimation of Word Representations in Vector Space"
- Hooktheory 数据集: https://www.hooktheory.com/theorytab
- Sheet Sage 项目: https://github.com/chrisdonahue/sheetsage

## 引用

如果使用此代码，请引用原始 chord2vec 工作和 Hooktheory 数据集来源（Sheet Sage）：

```bibtex
@inproceedings{donahue2022melody,
  title={Melody transcription via generative pre-training},
  author={Donahue, Chris and Thickstun, John and Liang, Percy},
  booktitle={ISMIR},
  year={2022}
}
```
