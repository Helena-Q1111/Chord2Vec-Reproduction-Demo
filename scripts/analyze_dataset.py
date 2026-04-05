"""
Analyze Hooktheory dataset structure and distribution.
"""

import gzip
import json
from collections import Counter
import numpy as np

# 加载数据
print("Loading Hooktheory dataset...")
with gzip.open('../Hooktheory.json.gz', 'rt', encoding='utf-8') as f:
    data = json.load(f)

print('='*60)
print('Hooktheory Dataset Overview')
print('='*60)
print(f'Total annotations: {len(data)}')

# 详细查看一个样本
sample_key = list(data.keys())[0]
sample = data[sample_key]

print('\n--- Sample Data Structure ---')
print(f'Entry ID: {sample_key}')
print(f'Tags: {sample.get("tags", [])}')
print(f'Split: {sample.get("split", "N/A")}')
print(f'YouTube ID: {sample.get("youtube", "N/A")}')
hooktheory_meta = sample.get('hooktheory', {})
print(f'Hooktheory meta keys: {list(hooktheory_meta.keys()) if hooktheory_meta else "None"}')

print('\n--- Annotations Structure ---')
annot = sample['annotations']
print(f'Num beats: {annot.get("num_beats", 0)}')
print(f'Meters: {annot.get("meters", [])}')
print(f'Keys: {annot.get("keys", [])}')
print(f'Melody entries: {len(annot.get("melody", []))}')
print(f'Harmony entries: {len(annot.get("harmony", []))}')

# 查看harmony结构
if annot.get('harmony'):
    print('\n--- Harmony Entry Example ---')
    h = annot['harmony'][0]
    for k, v in h.items():
        print(f'  {k}: {v}')

# 分析split分布
splits = Counter(item.get('split', 'none') for item in data.values())
print(f'\n--- Train/Valid/Test Split ---')
for split, count in splits.items():
    print(f'  {split}: {count} ({100*count/len(data):.1f}%)')

# 分析调性分布
all_keys = []
pitch_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
mode_counts = Counter()

for item in data.values():
    keys = item['annotations'].get('keys', [])
    for k in keys:
        tonic = k.get('tonic_pitch_class', 0)
        mode = k.get('mode', 'unknown')
        mode_counts[mode] += 1
        all_keys.append(f'{pitch_names[tonic]}_{mode}')

key_counts = Counter(all_keys)
print(f'\n--- Mode Distribution ---')
for mode, count in mode_counts.most_common():
    print(f'  {mode}: {count}')

print(f'\n--- Key Distribution (top 15) ---')
for key, count in key_counts.most_common(15):
    print(f'  {key}: {count}')

# 分析和弦类型分布
chord_qualities = Counter()
interval_patterns = Counter()

for item in data.values():
    harmony = item['annotations'].get('harmony', [])
    if harmony:
        for h in harmony:
            intervals = tuple(h.get('root_position_intervals', []))
            interval_patterns[intervals] += 1

# 映射interval到和弦类型
INTERVAL_TO_QUALITY = {
    (4, 3): 'Major',
    (3, 4): 'Minor',
    (3, 3): 'Diminished',
    (4, 4): 'Augmented',
    (4, 3, 4): 'Major 7th',
    (4, 3, 3): 'Dominant 7th',
    (3, 4, 3): 'Minor 7th',
    (3, 3, 3): 'Diminished 7th',
    (3, 3, 4): 'Half-dim 7th',
    (4, 4, 2): 'Augmented 7th',
    (3, 4, 4): 'Minor-Major 7th',
    (2, 5): 'Sus2',
    (5, 2): 'Sus4',
}

print(f'\n--- Chord Quality Distribution (top 20) ---')
for intervals, count in interval_patterns.most_common(20):
    quality = INTERVAL_TO_QUALITY.get(intervals, f'Other{intervals}')
    print(f'  {quality}: {count}')

# 分析和弦进行长度
seq_lengths = []
for item in data.values():
    harmony = item['annotations'].get('harmony', [])
    if harmony:
        seq_lengths.append(len(harmony))

print(f'\n--- Chord Sequence Length Statistics ---')
print(f'  Total sequences: {len(seq_lengths)}')
print(f'  Min length: {min(seq_lengths)}')
print(f'  Max length: {max(seq_lengths)}')
print(f'  Mean length: {np.mean(seq_lengths):.1f}')
print(f'  Median length: {np.median(seq_lengths):.1f}')

# 分析一些具体的hooktheory元数据
print('\n--- Hooktheory Metadata Sample ---')
count_with_meta = 0
artists = Counter()
for key, item in data.items():
    meta = item.get('hooktheory', {})
    if meta:
        count_with_meta += 1
        if 'artist' in meta:
            artists[meta['artist']] += 1

print(f'  Entries with Hooktheory metadata: {count_with_meta}')
print(f'  Unique artists: {len(artists)}')
print(f'\n  Top 20 artists:')
for artist, count in artists.most_common(20):
    print(f'    {artist}: {count}')

# 分析YouTube信息
youtube_count = sum(1 for item in data.values() if item.get('youtube'))
print(f'\n--- YouTube Info ---')
print(f'  Entries with YouTube ID: {youtube_count}')

# 查看更多hooktheory字段
print('\n--- Hooktheory Full Meta Example ---')
for key, item in data.items():
    meta = item.get('hooktheory', {})
    if meta:
        for k, v in meta.items():
            print(f'  {k}: {v}')
        break
