# Chord2Vec Demo

这个目录是一个纯前端可视化与听觉 demo，数据来自 `../output_ring_full_v4`。

## 内容

- 2D 散点图展示所有 chord 的 t-SNE 位置
- 点击 chord 播放默认钢琴音色
- 第二次点击会画出虚线并显示 cosine similarity
- 和弦点按类别着色：大和弦、小和弦、属和弦、增减和弦、其他

## 依赖

- 浏览器直接打开 `index.html`
- 需要联网加载 Tone.js 和钢琴采样
