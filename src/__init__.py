"""
Hateful Meme Detection Package

A multimodal content moderation system using CLIP with Cross-Attention Fusion
for detecting hateful content in internet memes.
"""

from .model import HatefulMemeClassifier, CrossAttentionFusion, create_model
from .inference import HatefulMemePredictor
from .losses import FocalLoss, get_loss_function
from .dataset import HatefulMemesDataset, create_dataloaders

__version__ = "1.0.0"
__author__ = "Bhrigu Anilkumar, Deepa Chandrasekar, Arshpreet Kaur"

__all__ = [
    'HatefulMemeClassifier',
    'CrossAttentionFusion',
    'create_model',
    'HatefulMemePredictor',
    'FocalLoss',
    'get_loss_function',
    'HatefulMemesDataset',
    'create_dataloaders'
]
