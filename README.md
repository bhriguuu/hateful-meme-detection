#  Multimodal Hateful Meme Detection

> **AI content moderation system** using CLIP with Cross-Attention Fusion for detecting hateful content in multimodal memes. Includes real-time Discord bot deployment.


---

## Problem Statement

Hateful memes represent a unique challenge in content moderation: **meaning emerges from the interaction between images and text**, not from either modality alone. A benign image becomes offensive when paired with certain text, and innocuous text transforms into hate speech with specific imagery.

Traditional content moderation systems analyze images and text separately, missing this critical semantic interplay. Our solution addresses this gap with a multimodal architecture that captures cross-modal relationships.

---

##  Key Features

| Feature | Description |
|---------|-------------|
| **Cross-Attention Fusion** | Bidirectional attention mechanism enabling joint image-text reasoning |
|  **Efficient Transfer Learning** | Only 3.8% trainable parameters (5.9M of 157M total) |
|  **Production Discord Bot** | Real-time content moderation with configurable thresholds |
|  **Comprehensive Evaluation** | ROC-AUC, PR curves, confusion matrices, error analysis |
| **Bias Auditing** | Built-in fairness evaluation and mitigation strategies |
|  **Focal Loss** | Handles class imbalance (64% vs 36%) effectively |
|  **Docker Ready** | Containerized deployment for production environments |

---

##  Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      HatefulMemeClassifier                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌──────────────┐                    ┌──────────────┐              │
│   │    Image     │                    │     Text     │              │
│   │    Input     │                    │    Input     │              │
│   └──────┬───────┘                    └──────┬───────┘              │
│          │                                   │                      │
│          ▼                                   ▼                      │
│   ┌──────────────┐                    ┌──────────────┐              │
│   │ CLIP Vision  │                    │  CLIP Text   │              │
│   │   Encoder    │                    │   Encoder    │              │
│   │  (Frozen)    │                    │  (Frozen)    │              │
│   └──────┬───────┘                    └──────┬───────┘              │
│          │ 512-dim                           │ 512-dim              │
│          ▼                                   ▼                      │
│   ┌──────────────┐                    ┌──────────────┐              │
│   │  Projection  │                    │  Projection  │              │
│   │    Layer     │                    │    Layer     │              │
│   └──────┬───────┘                    └──────┬───────┘              │
│          │                                   │                      │
│          └─────────────┬─────────────────────┘                      │
│                        │                                            │
│          ┌─────────────┴─────────────┐                              │
│          ▼                           ▼                              │
│   ┌──────────────┐            ┌──────────────┐                      │
│   │    Cross     │            │    Direct    │                      │
│   │  Attention   │            │    Concat    │                      │
│   │   Fusion     │            │    Fusion    │                      │
│   │  (8 heads)   │            │              │                      │
│   └──────┬───────┘            └──────┬───────┘                      │
│          │                           │                              │
│          └─────────────┬─────────────┘                              │
│                        │                                            │
│                        ▼                                            │
│               ┌──────────────┐                                      │
│               │ Classification│                                      │
│               │     Head      │                                      │
│               │  (MLP + BN)   │                                      │
│               └──────┬───────┘                                      │
│                      │                                              │
│                      ▼                                              │
│            [Hateful / Not Hateful]                                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Model Statistics

| Component | Parameters | Percentage |
|-----------|------------|------------|
| CLIP Encoders (Frozen) | 151,277,313 | 96.2% |
| Projection Layers | 1,050,624 | 0.7% |
| Cross-Attention | 2,101,248 | 1.3% |
| Classification Head | 2,763,265 | 1.8% |
| **Total Trainable** | **5,915,137** | **3.8%** |

---

## Performance

### Metrics on Validation Set

| Metric | Score |
|--------|-------|
| **Accuracy** | 61.00% |
| **F1 Score** | 0.6372 |
| **AUC-ROC** | 0.6761 |
| **Precision** | 0.6503 |
| **Recall** | 0.4760 |
| **Optimal Threshold** | 0.4274 |

### Context: This is a Hard Problem

| Model | Accuracy | Notes |
|-------|----------|-------|
| Human Performance | 84.7% | Gold standard |
| Competition Winners (VL-BERT, UNITER) | ~64.7% | Ensemble models |
| **Ours (Single Model)** | **61.0%** | Efficient, deployable |
| Random Baseline | 50.0% | - |

> The Hateful Memes Challenge is considered one of the hardest multimodal classification tasks. Our single-model approach achieves competitive results while being deployment-ready.

---

##  Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended, 8GB+ VRAM)
- 16GB+ RAM

### Installation

```bash
# Clone the repository
git clone https://github.com/bhriguuu/hateful-meme-detection.git
cd hateful-meme-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download model weights (if pre-trained)
python scripts/download_weights.py
```

### Single Image Inference

```python
from src.inference import HatefulMemePredictor

# Initialize predictor
predictor = HatefulMemePredictor(
    model_path="models/best_model.pth",
    device="cuda"  # or "cpu"
)

# Predict
result = predictor.predict(
    image_path="path/to/meme.jpg",
    text="Text overlay on the meme"
)

print(f"Prediction: {result['label']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Probability (hateful): {result['probability']:.4f}")
```

### Training

```bash
# Train from scratch
python src/train.py \
    --data_dir data/hateful_memes \
    --output_dir outputs/ \
    --epochs 10 \
    --batch_size 32 \
    --learning_rate 2e-4

# Resume training
python src/train.py \
    --resume outputs/checkpoint_epoch5.pth
```

### Discord Bot Deployment

```bash
# Set environment variables
export DISCORD_BOT_TOKEN="your_token_here"
export MONITORED_CHANNEL_ID="channel_id"

# Run the bot
python discord_bot/bot.py
```

---



##  Discord Bot

The trained model is deployed as a Discord bot for real-time content moderation.

### Threshold-Based Actions

| Probability | Action | Response |
|-------------|--------|----------|
| < 37% | ✅ **Safe** | Checkmark reaction |
| 37% - 42.74% | ⚠️ **Warning** | Warning reaction + advisory message |
| > 42.74% | 🛑 **Remove** | Auto-delete + notification to user |

### Bot Commands

```
!mod check   - Manually check the last image
!mod stats   - Show moderation statistics  
!mod info    - Display help and model info
```

### Setup Guide

1. Create Discord Application at [Discord Developer Portal](https://discord.com/developers/applications)
2. Enable **Message Content Intent** under Bot settings
3. Generate bot token
4. Invite bot with `Send Messages`, `Manage Messages`, `Add Reactions` permissions
5. Configure environment variables
6. Deploy (see [DEPLOYMENT.md](docs/DEPLOYMENT.md))

---

## Training Details

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Batch Size | 32 |
| Epochs | 10 (early stopping) |
| Learning Rate | 2e-4 |
| Optimizer | AdamW (β₁=0.9, β₂=0.999) |
| Scheduler | OneCycleLR (10% warmup) |
| Loss Function | Focal Loss (α=0.6412, γ=2.0) |
| Gradient Clipping | Max norm = 1.0 |
| Weight Decay | 0.01 |

### Data Augmentation

```python
# Training augmentations
- Resize(224, 224)
- HorizontalFlip(p=0.5)
- RandomBrightnessContrast(p=0.5)
- HueSaturationValue(p=0.3)
- GaussNoise(p=0.2)
- GaussianBlur(p=0.2)
- CLIP Normalization
```

---

##  Ethical Considerations

### Bias Analysis

We conducted comprehensive bias auditing identifying potential sources:

| Bias Type | Source | Mitigation |
|-----------|--------|------------|
| **Representation** | CLIP training data skews | Threshold optimization |
| **Annotation** | Subjective human labeling | Human-in-the-loop warnings |
| **Algorithmic** | Spurious correlations | Focal Loss, ensemble fusion |

### Fairness Metrics

| Metric | Value |
|--------|-------|
| Overall FPR | 0.256 |
| Overall TPR | 0.476 |

### Responsible Deployment

>  **Important**: This model achieves 61% accuracy and should be used as an **assistive tool with human oversight**, not as a standalone content moderation system.

**Recommended practices:**
- Always provide appeal mechanisms
- Log decisions for audit trails
- Regular bias audits on deployment data
- Human review for edge cases (warning zone)

---

##  Dataset

This project uses the **Facebook AI Hateful Memes Challenge** dataset.

| Split | Samples | Not Hateful | Hateful |
|-------|---------|-------------|---------|
| Train | 8,500 | 64.1% | 35.9% |
| Dev | 500 | 50.0% | 50.0% |
| Test | 1,000 | Hidden | Hidden |

### Accessing the Dataset

1. Visit [Facebook AI Hateful Memes Challenge](https://ai.meta.com/tools/hatefulmemes/)
2. Accept the license agreement
3. Download and extract to `data/hateful_memes/`

See [data/README.md](data/README.md) for detailed instructions.

---

## API Reference

### HatefulMemeClassifier

```python
from src.model import HatefulMemeClassifier

model = HatefulMemeClassifier(
    clip_model_name="openai/clip-vit-base-patch32",
    hidden_dim=512,
    num_heads=8,
    dropout=0.3,
    freeze_clip=True
)

# Forward pass
logits = model(pixel_values, input_ids, attention_mask)
```

### HatefulMemePredictor

```python
from src.inference import HatefulMemePredictor

predictor = HatefulMemePredictor(model_path, device="cuda")
result = predictor.predict(image_path, text)
result = predictor.predict_batch(image_paths, texts)
```

See [docs/API.md](docs/API.md) for complete API documentation.

---

## Docker Deployment

```bash
# Build image
docker build -t hateful-meme-detection .

# Run inference server
docker run -p 8000:8000 --gpus all hateful-meme-detection

# Run Discord bot
docker-compose up discord-bot
```

---

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test
pytest tests/test_model.py -v
```

---

### References

1. Kiela, D., et al. (2020). "The Hateful Memes Challenge: Detecting Hate Speech in Multimodal Memes." *NeurIPS*.
2. Radford, A., et al. (2021). "Learning Transferable Visual Models From Natural Language Supervision." *ICML*.
3. Lin, T.-Y., et al. (2017). "Focal Loss for Dense Object Detection." *ICCV*.

---

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---


## Acknowledgments

- Facebook AI for the Hateful Memes Challenge dataset
- OpenAI for the CLIP model
- Hugging Face for the Transformers library
- PyTorch team for the deep learning framework

---

<p align="center">
  <b>Built with ❤️ for safer online communities</b>
</p>
