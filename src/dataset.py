"""
Dataset module for Hateful Memes Challenge.

Provides PyTorch Dataset classes with:
- Image augmentation using Albumentations
- Text preprocessing
- CLIP-compatible input formatting
- Graceful handling of missing files
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Callable, Union

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import CLIPProcessor

try:
    import albumentations as A
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    print("Warning: albumentations not installed. Using basic transforms.")


# ============================================================================
# Image Augmentation Pipelines
# ============================================================================

def get_train_transform():
    """Get training augmentation pipeline."""
    if not ALBUMENTATIONS_AVAILABLE:
        return None
    
    return A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.2, 
            contrast_limit=0.2, 
            p=0.5
        ),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=10,
            p=0.3
        ),
        A.GaussNoise(var_limit=(10, 50), p=0.2),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],  # CLIP normalization
            std=[0.26862954, 0.26130258, 0.27577711]
        ),
    ])


def get_val_transform():
    """Get validation/test augmentation pipeline (no augmentation)."""
    if not ALBUMENTATIONS_AVAILABLE:
        return None
    
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        ),
    ])


# ============================================================================
# Text Preprocessing
# ============================================================================

def preprocess_text(text: Optional[str], max_length: int = 512) -> str:
    """
    Clean and preprocess text.
    
    Args:
        text: Input text string
        max_length: Maximum text length
        
    Returns:
        Cleaned text string
    """
    if not text or not isinstance(text, str):
        return "no text available"
    
    # Basic cleaning
    text = text.strip()
    
    # Handle empty text
    if not text:
        return "no text available"
    
    # Truncate very long texts
    if len(text) > max_length:
        text = text[:max_length]
    
    return text


# ============================================================================
# Dataset Classes
# ============================================================================

class HatefulMemesDataset(Dataset):
    """
    PyTorch Dataset for Hateful Memes Challenge.
    
    Features:
        - Image augmentation using Albumentations
        - Text preprocessing with CLIP tokenizer
        - Handles missing images gracefully
        - Returns CLIP-compatible inputs
    
    Args:
        data: List of dicts with 'img', 'text', 'label' keys
        data_path: Base path to dataset directory
        processor: CLIP processor for tokenization
        transform: Albumentations transform pipeline
        is_training: Whether this is training data
        max_text_length: Maximum text token length
    
    Example:
        >>> processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        >>> dataset = HatefulMemesDataset(
        ...     data=train_data,
        ...     data_path="data/hateful_memes",
        ...     processor=processor,
        ...     transform=get_train_transform()
        ... )
    """
    
    def __init__(
        self,
        data: List[Dict],
        data_path: Union[str, Path],
        processor: CLIPProcessor,
        transform: Optional[Callable] = None,
        is_training: bool = False,
        max_text_length: int = 77
    ):
        self.data = data
        self.data_path = Path(data_path)
        self.processor = processor
        self.transform = transform
        self.is_training = is_training
        self.max_text_length = max_text_length
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        # ============================================================
        # Load and process image
        # ============================================================
        img_path = self.data_path / item['img']
        
        try:
            image = Image.open(img_path).convert('RGB')
            image_np = np.array(image)
            
            # Apply augmentation
            if self.transform:
                augmented = self.transform(image=image_np)
                image_np = augmented['image']
            
            # Convert to tensor format [C, H, W]
            if isinstance(image_np, np.ndarray):
                image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float()
            else:
                image_tensor = image_np
                
        except Exception as e:
            # Fallback: create gray placeholder
            print(f"Warning: Could not load image {img_path}: {e}")
            image_tensor = torch.zeros(3, 224, 224)
        
        # ============================================================
        # Process text
        # ============================================================
        text = preprocess_text(item.get('text', ''))
        
        # Tokenize text using CLIP processor
        text_inputs = self.processor(
            text=[text],
            return_tensors="pt",
            padding='max_length',
            truncation=True,
            max_length=self.max_text_length
        )
        
        # ============================================================
        # Get label
        # ============================================================
        label = item.get('label', 0)
        
        return {
            'pixel_values': image_tensor,
            'input_ids': text_inputs['input_ids'].squeeze(0),
            'attention_mask': text_inputs['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.float),
            'text': text,
            'img_path': str(img_path),
            'id': item.get('id', idx)
        }


# ============================================================================
# Data Loading Utilities
# ============================================================================

def load_jsonl(filepath: Union[str, Path]) -> List[Dict]:
    """
    Load JSONL file into list of dicts.
    
    Args:
        filepath: Path to JSONL file
        
    Returns:
        List of dictionaries
    """
    data = []
    filepath = Path(filepath)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    
    return data


def create_dataloaders(
    data_path: Union[str, Path],
    processor: CLIPProcessor,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Dict[str, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        data_path: Path to dataset directory
        processor: CLIP processor
        batch_size: Batch size
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for GPU transfer
        
    Returns:
        Dictionary with 'train', 'val', 'test' dataloaders
    """
    data_path = Path(data_path)
    
    # Load data
    train_data = load_jsonl(data_path / 'train.jsonl')
    val_data = load_jsonl(data_path / 'dev.jsonl')
    test_data = load_jsonl(data_path / 'test.jsonl')
    
    # Create datasets
    train_dataset = HatefulMemesDataset(
        data=train_data,
        data_path=data_path,
        processor=processor,
        transform=get_train_transform(),
        is_training=True
    )
    
    val_dataset = HatefulMemesDataset(
        data=val_data,
        data_path=data_path,
        processor=processor,
        transform=get_val_transform(),
        is_training=False
    )
    
    test_dataset = HatefulMemesDataset(
        data=test_data,
        data_path=data_path,
        processor=processor,
        transform=get_val_transform(),
        is_training=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset
    }


def get_class_weights(data: List[Dict]) -> torch.Tensor:
    """
    Calculate class weights for imbalanced data.
    
    Args:
        data: List of data samples with 'label' key
        
    Returns:
        Tensor with class weights
    """
    labels = [item.get('label', 0) for item in data]
    n_samples = len(labels)
    n_positive = sum(labels)
    n_negative = n_samples - n_positive
    
    # Weight for positive class (minority)
    pos_weight = n_negative / n_positive if n_positive > 0 else 1.0
    
    return torch.tensor([pos_weight])


if __name__ == "__main__":
    # Test dataset loading
    from transformers import CLIPProcessor
    
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Test with sample data
    sample_data = [
        {'id': 1, 'img': 'img/test.png', 'text': 'sample text', 'label': 0}
    ]
    
    dataset = HatefulMemesDataset(
        data=sample_data,
        data_path='data/hateful_memes',
        processor=processor,
        transform=get_val_transform()
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Sample keys: {list(dataset[0].keys())}")
