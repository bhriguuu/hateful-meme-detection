# API Reference

Complete API documentation for the Hateful Meme Detection system.

## Table of Contents

- [Model Classes](#model-classes)
- [Inference API](#inference-api)
- [Dataset Classes](#dataset-classes)
- [Loss Functions](#loss-functions)
- [Utility Functions](#utility-functions)

---

## Model Classes

### `HatefulMemeClassifier`

Main classifier model using CLIP with Cross-Attention Fusion.

```python
from src.model import HatefulMemeClassifier

model = HatefulMemeClassifier(
    clip_model_name: str = "openai/clip-vit-base-patch32",
    hidden_dim: int = 512,
    num_heads: int = 8,
    dropout: float = 0.3,
    freeze_clip: bool = True
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `clip_model_name` | str | "openai/clip-vit-base-patch32" | HuggingFace CLIP model identifier |
| `hidden_dim` | int | 512 | Hidden dimension for fusion layers |
| `num_heads` | int | 8 | Number of attention heads |
| `dropout` | float | 0.3 | Dropout probability |
| `freeze_clip` | bool | True | Whether to freeze CLIP encoder weights |

#### Methods

##### `forward(pixel_values, input_ids, attention_mask) -> Tensor`

Forward pass through the model.

```python
logits = model(pixel_values, input_ids, attention_mask)
probabilities = torch.sigmoid(logits)
```

**Arguments:**
- `pixel_values` (Tensor): Preprocessed images `[batch, 3, 224, 224]`
- `input_ids` (Tensor): Tokenized text `[batch, seq_len]`
- `attention_mask` (Tensor): Attention mask `[batch, seq_len]`

**Returns:**
- `Tensor`: Logits `[batch]` - apply sigmoid for probabilities

##### `predict(pixel_values, input_ids, attention_mask, threshold) -> Tuple[Tensor, Tensor]`

Make predictions with threshold-based classification.

```python
predictions, probabilities = model.predict(
    pixel_values, input_ids, attention_mask,
    threshold=0.4274
)
```

##### `count_parameters() -> Dict[str, int]`

Count model parameters.

```python
params = model.count_parameters()
# {'total': 157192450, 'trainable': 5915137, 'frozen': 151277313, 'trainable_pct': 3.76}
```

---

### `CrossAttentionFusion`

Cross-attention module for bidirectional image-text fusion.

```python
from src.model import CrossAttentionFusion

fusion = CrossAttentionFusion(
    embed_dim: int,
    num_heads: int = 8,
    dropout: float = 0.1
)
```

#### Methods

##### `forward(img_feat, txt_feat) -> Tensor`

Fuse image and text features.

```python
fused = fusion(img_feat, txt_feat)  # [batch, embed_dim]
```

---

## Inference API

### `HatefulMemePredictor`

Production-ready predictor class with preprocessing built-in.

```python
from src.inference import HatefulMemePredictor

predictor = HatefulMemePredictor(
    model_path: str,
    device: str = "cuda",
    threshold: float = 0.4274
)
```

#### Methods

##### `predict(image, text, return_logits) -> Dict`

Predict for single image-text pair.

```python
result = predictor.predict(
    image="path/to/meme.jpg",  # or PIL.Image or np.ndarray
    text="text on the meme",
    return_logits=False
)

# Returns:
{
    'is_hateful': False,
    'label': 'NOT HATEFUL',
    'probability': 0.2341,
    'confidence': 0.7659,
    'threshold': 0.4274
}
```

##### `predict_batch(images, texts, batch_size) -> List[Dict]`

Batch prediction for multiple memes.

```python
results = predictor.predict_batch(
    images=["meme1.jpg", "meme2.jpg", "meme3.jpg"],
    texts=["text 1", "text 2", "text 3"],
    batch_size=16
)
```

##### `get_detailed_analysis(image, text) -> Dict`

Get detailed analysis with zone classification.

```python
analysis = predictor.get_detailed_analysis(image, text)

# Returns:
{
    'is_hateful': False,
    'label': 'NOT HATEFUL',
    'probability': 0.2341,
    'confidence': 0.7659,
    'threshold': 0.4274,
    'zone': 'SAFE',  # 'SAFE', 'WARNING', or 'HARMFUL'
    'zone_color': 'green',
    'text_analyzed': 'text on the meme...',
    'model_info': {...}
}
```

##### `set_threshold(threshold: float)`

Update classification threshold.

```python
predictor.set_threshold(0.5)  # More lenient
```

---

## Dataset Classes

### `HatefulMemesDataset`

PyTorch Dataset for loading hateful memes data.

```python
from src.dataset import HatefulMemesDataset

dataset = HatefulMemesDataset(
    data: List[Dict],
    data_path: str,
    processor: CLIPProcessor,
    transform: Optional[Callable] = None,
    is_training: bool = False,
    max_text_length: int = 77
)
```

#### Item Format

```python
item = dataset[0]
# {
#     'pixel_values': Tensor[3, 224, 224],
#     'input_ids': Tensor[77],
#     'attention_mask': Tensor[77],
#     'label': Tensor (float),
#     'text': str,
#     'img_path': str,
#     'id': int
# }
```

### Helper Functions

##### `create_dataloaders(data_path, processor, batch_size, num_workers) -> Dict`

Create train/val/test dataloaders.

```python
from src.dataset import create_dataloaders

loaders = create_dataloaders(
    data_path="data/hateful_memes",
    processor=processor,
    batch_size=32,
    num_workers=4
)

train_loader = loaders['train']
val_loader = loaders['val']
test_loader = loaders['test']
```

##### `load_jsonl(filepath) -> List[Dict]`

Load JSONL file.

```python
from src.dataset import load_jsonl

data = load_jsonl("data/hateful_memes/train.jsonl")
```

##### `get_train_transform() -> A.Compose`

Get training augmentation pipeline.

##### `get_val_transform() -> A.Compose`

Get validation/test transform pipeline.

---

## Loss Functions

### `FocalLoss`

Focal Loss for handling class imbalance.

```python
from src.losses import FocalLoss

criterion = FocalLoss(
    alpha: float = 0.25,  # Weight for positive class
    gamma: float = 2.0,   # Focusing parameter
    reduction: str = 'mean'
)

loss = criterion(logits, targets)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `alpha` | float | 0.25 | Weighting factor for positive class |
| `gamma` | float | 2.0 | Focusing parameter (0 = BCE) |
| `reduction` | str | 'mean' | 'mean', 'sum', or 'none' |

### `get_loss_function`

Factory function for creating loss functions.

```python
from src.losses import get_loss_function

# Focal Loss
criterion = get_loss_function('focal', alpha=0.6412, gamma=2.0)

# Weighted BCE
criterion = get_loss_function('weighted_bce', pos_weight=1.78)

# Label Smoothing BCE
criterion = get_loss_function('smoothed_bce', smoothing=0.1)
```

---

## Utility Functions

### Metrics

```python
from src.utils import calculate_metrics, find_optimal_threshold

# Calculate all metrics
metrics = calculate_metrics(y_true, y_pred, y_prob)
# {'accuracy': 0.61, 'f1': 0.637, 'precision': 0.65, 'recall': 0.476, ...}

# Find optimal threshold
threshold, score = find_optimal_threshold(y_true, y_prob, metric='f1')
```

### Model Utilities

```python
from src.utils import count_parameters, get_device, save_checkpoint, load_checkpoint

# Get best device
device = get_device(prefer_gpu=True)

# Count parameters
params = count_parameters(model)

# Save checkpoint
save_checkpoint(model, optimizer, epoch, metrics, 'checkpoint.pth', config)

# Load checkpoint
checkpoint = load_checkpoint('checkpoint.pth', model, optimizer, device)
```

### Configuration

```python
from src.utils import load_config, save_config, merge_configs

# Load config
config = load_config('configs/model_config.json')

# Save config
save_config(config, 'outputs/config.json')

# Merge configs
config = merge_configs(default_config, user_config)
```

### Logging

```python
from src.utils import setup_logging

logger = setup_logging(
    log_dir='logs/',
    log_level=logging.INFO
)

logger.info("Training started")
```

---

## Usage Examples

### Complete Inference Pipeline

```python
from src.inference import HatefulMemePredictor

# Initialize
predictor = HatefulMemePredictor(
    model_path="models/best_model.pth",
    device="cuda"
)

# Single prediction
result = predictor.predict(
    image="meme.jpg",
    text="some text"
)

if result['is_hateful']:
    print(f"⚠️ Hateful content detected ({result['probability']:.1%})")
else:
    print(f"✅ Content appears safe ({result['confidence']:.1%} confident)")
```

### Training Loop

```python
from src.model import HatefulMemeClassifier
from src.dataset import create_dataloaders
from src.losses import FocalLoss
from transformers import CLIPProcessor

# Setup
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
loaders = create_dataloaders("data/hateful_memes", processor, batch_size=32)
model = HatefulMemeClassifier().cuda()
criterion = FocalLoss(alpha=0.6412, gamma=2.0)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

# Train
for batch in loaders['train']:
    optimizer.zero_grad()
    outputs = model(
        batch['pixel_values'].cuda(),
        batch['input_ids'].cuda(),
        batch['attention_mask'].cuda()
    )
    loss = criterion(outputs, batch['label'].cuda())
    loss.backward()
    optimizer.step()
```

---

## Error Handling

All API functions include proper error handling:

```python
try:
    result = predictor.predict(image_path, text)
except FileNotFoundError:
    print("Image not found")
except RuntimeError as e:
    print(f"Model error: {e}")
```

## Type Hints

All functions include complete type hints for IDE support:

```python
def predict(
    self,
    image: Union[str, Path, Image.Image, np.ndarray],
    text: str,
    return_logits: bool = False
) -> Dict[str, Any]:
    ...
```
