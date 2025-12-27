"""
Utility functions for Hateful Meme Detection.

Provides helper functions for:
- Logging setup
- Configuration management
- Metrics calculation
- Visualization
- File I/O operations
"""

import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


# ============================================================================
# Logging
# ============================================================================

def setup_logging(
    log_dir: Optional[Union[str, Path]] = None,
    log_level: int = logging.INFO,
    log_name: str = "hateful_meme_detection"
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_dir: Directory for log files (None for console only)
        log_level: Logging level
        log_name: Logger name
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(log_name)
    logger.setLevel(log_level)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Format
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_dir specified)
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'{log_name}_{timestamp}.log'
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to: {log_file}")
    
    return logger


# ============================================================================
# Reproducibility
# ============================================================================

def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)


# ============================================================================
# Configuration Management
# ============================================================================

def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config


def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """
    Save configuration to JSON file.
    
    Args:
        config: Configuration dictionary
        config_path: Output path
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries.
    Later configs override earlier ones.
    
    Args:
        *configs: Configuration dictionaries
        
    Returns:
        Merged configuration
    """
    result = {}
    for config in configs:
        result.update(config)
    return result


# ============================================================================
# Metrics Calculation
# ============================================================================

def calculate_metrics(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray],
    y_prob: Optional[Union[List, np.ndarray]] = None
) -> Dict[str, float]:
    """
    Calculate comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities (optional)
        
    Returns:
        Dictionary of metrics
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
    }
    
    # AUC requires probabilities
    if y_prob is not None:
        y_prob = np.array(y_prob)
        try:
            metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
            
            # Average precision
            precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob)
            metrics['avg_precision'] = auc(recall_vals, precision_vals)
        except ValueError:
            metrics['auc_roc'] = 0.0
            metrics['avg_precision'] = 0.0
    
    # Confusion matrix values
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics.update({
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
    })
    
    return metrics


def find_optimal_threshold(
    y_true: Union[List, np.ndarray],
    y_prob: Union[List, np.ndarray],
    metric: str = 'f1'
) -> Tuple[float, float]:
    """
    Find optimal classification threshold.
    
    Args:
        y_true: True labels
        y_prob: Prediction probabilities
        metric: Metric to optimize ('f1', 'youden', 'balanced')
        
    Returns:
        Tuple of (optimal_threshold, optimal_score)
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    thresholds = np.linspace(0.1, 0.9, 100)
    best_threshold = 0.5
    best_score = 0.0
    
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        
        if metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'youden':
            # Youden's J statistic: sensitivity + specificity - 1
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            score = sensitivity + specificity - 1
        elif metric == 'balanced':
            # Balanced accuracy
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            score = (sensitivity + specificity) / 2
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        if score > best_score:
            best_score = score
            best_threshold = thresh
    
    return best_threshold, best_score


# ============================================================================
# Model Utilities
# ============================================================================

def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total,
        'trainable': trainable,
        'frozen': total - trainable,
        'trainable_pct': trainable / total * 100 if total > 0 else 0
    }


def get_device(prefer_gpu: bool = True) -> torch.device:
    """
    Get best available device.
    
    Args:
        prefer_gpu: Whether to prefer GPU if available
        
    Returns:
        torch.device
    """
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    metrics: Dict[str, float],
    path: Union[str, Path],
    config: Optional[Dict] = None
) -> None:
    """
    Save training checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state (optional)
        epoch: Current epoch
        metrics: Training metrics
        path: Save path
        config: Model configuration
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if config is not None:
        checkpoint['model_config'] = config
    
    torch.save(checkpoint, path)


def load_checkpoint(
    path: Union[str, Path],
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Load training checkpoint.
    
    Args:
        path: Checkpoint path
        model: Model to load weights into
        optimizer: Optimizer to load state into
        device: Device to load on
        
    Returns:
        Checkpoint dictionary
    """
    checkpoint = torch.load(path, map_location=device)
    
    if model is not None and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


# ============================================================================
# File Utilities
# ============================================================================

def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if not.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_timestamp() -> str:
    """Get current timestamp string."""
    return datetime.now().strftime('%Y%m%d_%H%M%S')


# ============================================================================
# Text Utilities
# ============================================================================

def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to max length.
    
    Args:
        text: Input text
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def clean_text(text: str) -> str:
    """
    Clean and normalize text.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Strip whitespace
    text = text.strip()
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    return text


if __name__ == "__main__":
    # Test utilities
    print("Testing utilities...")
    
    # Test seed setting
    set_seed(42)
    print("✓ Seed set")
    
    # Test metrics
    y_true = [0, 0, 1, 1, 0, 1]
    y_pred = [0, 1, 1, 1, 0, 0]
    y_prob = [0.2, 0.6, 0.8, 0.9, 0.3, 0.4]
    
    metrics = calculate_metrics(y_true, y_pred, y_prob)
    print(f"✓ Metrics calculated: F1={metrics['f1']:.3f}")
    
    # Test threshold finding
    threshold, score = find_optimal_threshold(y_true, y_prob)
    print(f"✓ Optimal threshold: {threshold:.3f} (F1={score:.3f})")
    
    print("\nAll utilities working correctly!")
