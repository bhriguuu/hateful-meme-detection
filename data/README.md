# Dataset Setup

This directory should contain the **Facebook AI Hateful Memes Challenge** dataset.

## Dataset Information

| Split | Samples | Not Hateful | Hateful |
|-------|---------|-------------|---------|
| Train | 8,500 | 64.1% (5,450) | 35.9% (3,050) |
| Dev | 500 | 50.0% (250) | 50.0% (250) |
| Test | 1,000 | Hidden | Hidden |

## Download Instructions

### Option 1: From Facebook AI (Official)

1. Visit: [Facebook AI Hateful Memes Challenge](https://ai.meta.com/tools/hatefulmemes/)
2. Accept the license agreement
3. Download the dataset
4. Extract to this directory

### Option 2: From Kaggle

```bash
# Install Kaggle CLI
pip install kaggle

# Configure Kaggle API (put kaggle.json in ~/.kaggle/)
kaggle competitions download -c hateful-memes

# Extract
unzip hateful-memes.zip -d hateful_memes/
```

### Option 3: From DrivenData

1. Visit: [DrivenData Hateful Memes](https://www.drivendata.org/competitions/64/hateful-memes/)
2. Register and accept terms
3. Download from the data section

## Expected Structure

After download, your `data/` directory should look like:

```
data/
├── README.md (this file)
└── hateful_memes/
    ├── train.jsonl
    ├── dev.jsonl
    ├── test.jsonl
    ├── LICENSE.txt
    ├── README.md
    └── img/
        ├── 00001.png
        ├── 00002.png
        ├── ...
        └── XXXXX.png
```

## Data Format

Each JSONL file contains lines with the following format:

```json
{
  "id": 42953,
  "img": "img/42953.png",
  "label": 0,
  "text": "its their character not their color that matters"
}
```

**Fields:**
- `id`: Unique identifier
- `img`: Relative path to image file
- `label`: 0 (not hateful) or 1 (hateful) - test set has no labels
- `text`: Text content of the meme

## Verification Script

Run this to verify your data setup:

```python
import os
import json

data_path = "data/hateful_memes"

# Check files exist
files = ['train.jsonl', 'dev.jsonl', 'test.jsonl']
for f in files:
    path = os.path.join(data_path, f)
    if os.path.exists(path):
        with open(path) as file:
            count = sum(1 for _ in file)
        print(f"✅ {f}: {count} samples")
    else:
        print(f"❌ {f}: NOT FOUND")

# Check images
img_path = os.path.join(data_path, "img")
if os.path.exists(img_path):
    img_count = len([f for f in os.listdir(img_path) if f.endswith('.png')])
    print(f"✅ Images: {img_count} files")
else:
    print("❌ img/ directory NOT FOUND")
```

## License

The Hateful Memes dataset is released under a custom license by Facebook AI.
Please read and comply with the terms in `LICENSE.txt`.

**Important:** This dataset is for research purposes only. Do not redistribute.

## Citation

If you use this dataset, please cite:

```bibtex
@inproceedings{kiela2020hateful,
  title={The Hateful Memes Challenge: Detecting Hate Speech in Multimodal Memes},
  author={Kiela, Douwe and Firooz, Hamed and Mober, Aravind and Goswami, Vedanuj and Shekhar, Sanjay and Wang, Xinyi and Prikhodko, Dmitry and Mazzini, Filippo and Bhattacharjee, Arnav and Alpert, Barbara and others},
  booktitle={NeurIPS},
  year={2020}
}
```
