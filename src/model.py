"""
Multimodal Hateful Meme Detection Model

CLIP-based multimodal classifier with Cross-Attention Fusion for detecting
hateful content in internet memes.

Architecture:
    - Frozen CLIP Vision/Text Encoders (ViT-B/32)
    - Trainable projection layers
    - Cross-Attention Fusion module
    - Concatenation Fusion (ensemble)
    - Deep classification head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel
from typing import Optional, Tuple, Dict


class CrossAttentionFusion(nn.Module):
    """
    Cross-Attention module for bidirectional image-text fusion.
    
    Allows each modality to attend to the other, capturing the semantic
    interplay that determines whether a meme is hateful.
    
    Args:
        embed_dim: Dimension of embeddings
        num_heads: Number of attention heads
        dropout: Dropout probability
    """
    
    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int = 8, 
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Image attends to text
        self.img_to_txt_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Text attends to image
        self.txt_to_img_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(embed_dim)
    
    def forward(
        self, 
        img_feat: torch.Tensor, 
        txt_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of cross-attention fusion.
        
        Args:
            img_feat: Image features [batch, embed_dim]
            txt_feat: Text features [batch, embed_dim]
            
        Returns:
            Fused features [batch, embed_dim]
        """
        # Add sequence dimension for attention
        img_feat = img_feat.unsqueeze(1)  # [batch, 1, embed_dim]
        txt_feat = txt_feat.unsqueeze(1)  # [batch, 1, embed_dim]
        
        # Cross attention: image attends to text
        img_attended, _ = self.img_to_txt_attn(img_feat, txt_feat, txt_feat)
        img_feat = self.norm1(img_feat + img_attended)
        
        # Cross attention: text attends to image
        txt_attended, _ = self.txt_to_img_attn(txt_feat, img_feat, img_feat)
        txt_feat = self.norm2(txt_feat + txt_attended)
        
        # Combine attended features
        combined = img_feat + txt_feat  # [batch, 1, embed_dim]
        
        # Feed-forward with residual
        combined = self.norm3(combined + self.ffn(combined))
        
        return combined.squeeze(1)  # [batch, embed_dim]


class HatefulMemeClassifier(nn.Module):
    """
    Advanced Multimodal Classifier for Hateful Meme Detection.
    
    Uses frozen CLIP encoders with trainable fusion layers and classification
    head. Combines cross-attention fusion with direct concatenation for
    robust multimodal reasoning.
    
    Args:
        clip_model_name: HuggingFace CLIP model identifier
        hidden_dim: Hidden dimension for fusion layers
        num_heads: Number of attention heads in cross-attention
        dropout: Dropout probability
        freeze_clip: Whether to freeze CLIP encoder parameters
    
    Example:
        >>> model = HatefulMemeClassifier()
        >>> logits = model(pixel_values, input_ids, attention_mask)
        >>> probs = torch.sigmoid(logits)
    """
    
    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.3,
        freeze_clip: bool = True
    ):
        super().__init__()
        
        self.clip_model_name = clip_model_name
        self.hidden_dim = hidden_dim
        
        # Load CLIP model
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        self.embed_dim = self.clip.config.projection_dim  # 512 for ViT-B/32
        
        # Freeze CLIP parameters for transfer learning
        if freeze_clip:
            for param in self.clip.parameters():
                param.requires_grad = False
        
        # Projection layers for dimension alignment
        self.img_proj = nn.Sequential(
            nn.Linear(self.embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.txt_proj = nn.Sequential(
            nn.Linear(self.embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Cross-attention fusion module
        self.cross_attention = CrossAttentionFusion(
            hidden_dim, num_heads, dropout
        )
        
        # Direct concatenation fusion (ensemble approach)
        self.concat_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Classification head
        # Input: cross-attention output + concatenation output
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform."""
        for module in [self.img_proj, self.txt_proj, 
                       self.concat_proj, self.classifier]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
    
    def get_clip_features(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract features from frozen CLIP encoders.
        
        Args:
            pixel_values: Preprocessed images [batch, 3, 224, 224]
            input_ids: Tokenized text [batch, seq_len]
            attention_mask: Text attention mask [batch, seq_len]
            
        Returns:
            Tuple of (image_features, text_features)
        """
        image_features = self.clip.get_image_features(pixel_values=pixel_values)
        text_features = self.clip.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return image_features, text_features
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the classifier.
        
        Args:
            pixel_values: Preprocessed images [batch, 3, 224, 224]
            input_ids: Tokenized text [batch, seq_len]
            attention_mask: Text attention mask [batch, seq_len]
            
        Returns:
            Logits [batch] - use sigmoid for probabilities
        """
        # Get CLIP embeddings
        image_features, text_features = self.get_clip_features(
            pixel_values, input_ids, attention_mask
        )
        
        # Normalize features
        image_features = F.normalize(image_features, p=2, dim=-1)
        text_features = F.normalize(text_features, p=2, dim=-1)
        
        # Project to hidden dimension
        img_proj = self.img_proj(image_features)
        txt_proj = self.txt_proj(text_features)
        
        # Cross-attention fusion
        cross_attn_out = self.cross_attention(img_proj, txt_proj)
        
        # Concatenation fusion
        concat_out = self.concat_proj(
            torch.cat([img_proj, txt_proj], dim=-1)
        )
        
        # Combine both fusion methods
        combined = torch.cat([cross_attn_out, concat_out], dim=-1)
        
        # Classification
        logits = self.classifier(combined)
        
        return logits.squeeze(-1)
    
    def predict(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        threshold: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with probability outputs.
        
        Args:
            pixel_values: Preprocessed images
            input_ids: Tokenized text
            attention_mask: Text attention mask
            threshold: Classification threshold
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(pixel_values, input_ids, attention_mask)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()
        return preds, probs
    
    def count_parameters(self) -> Dict[str, int]:
        """Count model parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        
        return {
            'total': total,
            'trainable': trainable,
            'frozen': frozen,
            'trainable_pct': trainable / total * 100
        }


def create_model(
    config: Optional[Dict] = None,
    pretrained_path: Optional[str] = None,
    device: str = "cuda"
) -> HatefulMemeClassifier:
    """
    Factory function to create model instance.
    
    Args:
        config: Model configuration dictionary
        pretrained_path: Path to pretrained weights
        device: Device to load model on
        
    Returns:
        Initialized HatefulMemeClassifier
    """
    default_config = {
        'clip_model_name': 'openai/clip-vit-base-patch32',
        'hidden_dim': 512,
        'num_heads': 8,
        'dropout': 0.3,
        'freeze_clip': True
    }
    
    if config:
        default_config.update(config)
    
    model = HatefulMemeClassifier(**default_config)
    
    if pretrained_path:
        checkpoint = torch.load(pretrained_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    
    return model.to(device)


if __name__ == "__main__":
    # Test model creation
    model = HatefulMemeClassifier()
    params = model.count_parameters()
    
    print("HatefulMemeClassifier")
    print("=" * 50)
    print(f"Total parameters:     {params['total']:,}")
    print(f"Trainable parameters: {params['trainable']:,}")
    print(f"Frozen parameters:    {params['frozen']:,}")
    print(f"Trainable percentage: {params['trainable_pct']:.2f}%")
