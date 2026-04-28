# IRIS Model: Tokenizer + WorldModel Training Pipeline

提取自 [Transformers are Sample-Efficient World Models (IRIS)](https://github.com/自动驾驶小鱼/iris) 的核心网络架构，在 DROID 数据集上训练。

---

## 项目结构

```
iris_model/
├── src/
│   ├── __init__.py
│   ├── batch.py              # Batch dataclass
│   ├── utils.py              # Loss utilities, init_weights
│   ├── models/
│   │   ├── __init__.py
│   │   ├── tokenizer.py      # VQ-VAE Tokenizer (离散自编码器)
│   │   ├── nets.py           # Encoder/Decoder (VQGAN-style ResNet+Attn)
│   │   ├── lpips.py          # Learned Perceptual Image Patch Similarity
│   │   ├── transformer.py     # GPT-style autoregressive Transformer
│   │   ├── slicer.py         # Token slicing utilities
│   │   ├── kv_caching.py     # KV Cache for efficient inference
│   │   └── world_model.py     # Autoregressive World Model
│   └── data/
│       ├── __init__.py
│       └── droid_dataset.py   # DROID-LeRobot dataset loader
├── train.py                   # Joint training loop
├── validate.py                 # Retrieval evaluation (Hit@1/5/10)
├── inference.py               # Inference + retrieval
├── retrieval.py               # Retrieval evaluation module
├── evaluator_retrieval_ce.py  # CE-based per-episode retrieval evaluation
├── run_train.sh
├── run_validate.sh
├── run_inference.sh
└── requirements.txt
```

---

## 模型架构

### Tokenizer (离散 VQ-VAE)

- **输入**: 图像 (B, 3, H, W) ∈ [0, 1]
- **Encoder**: VQGAN-style encoder with ResNet blocks + self-attention
- **Codebook**: `vocab_size=1024`, `embed_dim=512`
- **输出**: 离散 token 序列 (B, h*w) 其中 h=w=8 (latent resolution), K_obs=64
- **损失**: L1 重构损失 + LPIPS 感知损失 + Commitment 损失

### WorldModel (自回归 Transformer)

- **输入**: 交错的 observation tokens + action tokens
- **Token 布局**: `[obs_0, ..., obs_{K-1}, act_0, obs_0, ..., obs_{K-1}, act_1, ...]`
- **预测目标**: 下一个 observation token (仅 obs prediction, 不预测 reward / end)
- **损失**: Cross-Entropy on obs token prediction
- **配置**: 10层 Transformer, 4头, embed_dim=256, causal attention, max_blocks=20

---

## 数据说明

| 项目 | 说明 |
|---|---|
| 数据集 | DROID-LeRobot `/data3/v-jepa/jepa-wms/dataset/droid_lerobot_dataset` |
| 划分文件 | `/data3/v-jepa/jepa-wms/1w2splits.json` |
| 训练集 | splits["train"] 中的 episode (12,000 episodes) |
| 验证集 | splits["val"] 中的 episode (806 episodes) |
| 测试集 | splits["test"] 中的 episode (800 episodes) |
| 相机 | 训练时随机选择 exterior_image_1_left 或 exterior_image_2_left |
| 相机 | 评估时固定使用 exterior_image_1_left |
| 采样步长 | `stride=5`, 即每5帧取一帧 |
| Action | `action[t] = sampled_state[t+1] - sampled_state[t]` (stride 采样后的状态差分) |
| State | `observation.state`: 8维 (x, y, z, roll, pitch, yaw, gripper, pad), pad 维在加载时去掉 |
| 图像尺寸 | 256×256 |

---

## 运行方式

### 1. 安装依赖

```bash
cd iris_model
pip install -r requirements.txt
```

### 2. 训练

```bash
bash run_train.sh
# 或手动指定参数:
python train.py \
    --exp_dir ./experiments/run_001 \
    --img_size 256 \
    --batch_size 16 \
    --epochs 50 \
    --lr_tokenizer 3e-4 \
    --lr_worldmodel 1e-4 \
    --vocab_size 1024 \
    --embed_dim 512 \
    --device cuda
```

- Stage 1: 训练 Tokenizer, 每 2 epoch 保存一次 → `tokenizer_epoch_XXX.pth`
- Stage 2: 训练 WorldModel (Tokenizer 冻结), 每 5 epoch 保存一次 → `worldmodel_epoch_XXX.pth`
- 权重分开保存: tokenizer checkpoint 只含 tokenizer, worldmodel checkpoint 只含 worldmodel

### 3. 检索评估 (CE-based, per-episode)

```bash
conda activate iris_env

python evaluator_retrieval_ce.py \
    --tokenizer_checkpoint experiments/run_001/tokenizer_final.pth \
    --worldmodel_checkpoint experiments/run_001/worldmodel_epoch_050.pth \
    --device cuda:1 \
    --output experiments/run_001/iris_retrieval_results.txt
```

输出格式 (`iris_retrieval_results.txt`):

```
IRIS Retrieval Evaluation (per-episode, CE Loss)
Stride: 5 | ImgSize: 256 | MaxFrames: 20
================================================================================
Ep_000042_Q1    Top10=[3(0.1234) | 5(0.2345) | ...]    GT=3(0.1234)
...
================================================================================
Summary | Episodes: 806 | Samples: 15234 | Hit@1: 0.1234 | Hit@5: 0.4567 | Hit@10: 0.7890
```

### 4. 验证 / 推理 (旧脚本)

```bash
bash run_validate.sh
# 或:
python validate.py \
    --checkpoint experiments/run_001/checkpoint_epoch_050.pth \
    --exp_dir experiments/run_001 \
    --img_size 256 \
    --stride 5 \
    --seq_len 5 \
    --output l1_results.txt
```

---

## 检索逻辑 (evaluator_retrieval_ce.py)

两阶段设计 (与 jepa_wms evaluator 结构一致):

### Phase 1: 特征收集

对每个 val episode:
1. 加载全部采样帧 (stride=5)
2. Tokenizer 编码所有帧 → 离散 tokens (Key 池, shape `(N, K)`)
3. WorldModel 一次前向 → 每个 block t 预测 frame t+1 的 logits (Query, shape `(K, vocab)`)

### Phase 2: Episode 内检索

对每个 query position t (共 N-1 个):
1. 计算 CE(pred_logits_t, key_tokens_i) 与所有 key 帧 (向量化)
2. 按 CE 升序排列 (CE 越小 = 匹配越好)
3. 找 GT frame (t+1) 的 rank → 判定 Hit@k

### 关键参数

| 参数 | 说明 |
|------|------|
| Key | Tokenizer 编码的离散 token 索引 `(K,)` |
| Query | WorldModel 预测的 logits `(K, vocab)` |
| 距离 | Cross-Entropy loss (与训练 loss 一致) |
| 检索范围 | 同一 episode 内部, 无 group |
| 数据集 | 1w2splits.json 的 val 集 (806 episodes) |
| 最大帧数 | `wm_max_blocks=20` (Transformer 上下文长度限制) |
| 每个 episode 的 query 数 | N-1 (N 为该 episode 采样后的帧数, 最多 19) |
