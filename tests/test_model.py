"""
Unit tests for Hateful Meme Detection model.

Run with: pytest tests/test_model.py -v
"""

import pytest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


class TestCrossAttentionFusion:
    """Tests for CrossAttentionFusion module."""
    
    def test_init(self):
        """Test module initialization."""
        from model import CrossAttentionFusion
        
        module = CrossAttentionFusion(embed_dim=512, num_heads=8)
        assert module.embed_dim == 512
    
    def test_forward_shape(self):
        """Test output shape."""
        from model import CrossAttentionFusion
        
        module = CrossAttentionFusion(embed_dim=512, num_heads=8)
        
        batch_size = 4
        img_feat = torch.randn(batch_size, 512)
        txt_feat = torch.randn(batch_size, 512)
        
        output = module(img_feat, txt_feat)
        
        assert output.shape == (batch_size, 512)
    
    def test_different_embed_dims(self):
        """Test with different embedding dimensions."""
        from model import CrossAttentionFusion
        
        for dim in [256, 512, 768]:
            module = CrossAttentionFusion(embed_dim=dim, num_heads=8)
            img_feat = torch.randn(2, dim)
            txt_feat = torch.randn(2, dim)
            output = module(img_feat, txt_feat)
            assert output.shape == (2, dim)


class TestHatefulMemeClassifier:
    """Tests for HatefulMemeClassifier model."""
    
    @pytest.fixture
    def model(self):
        """Create model fixture (skip if no GPU and takes too long)."""
        from model import HatefulMemeClassifier
        
        # Use smaller hidden dim for faster testing
        model = HatefulMemeClassifier(
            hidden_dim=256,
            num_heads=4,
            dropout=0.1
        )
        model.eval()
        return model
    
    def test_model_init(self, model):
        """Test model initialization."""
        assert model is not None
        assert hasattr(model, 'clip')
        assert hasattr(model, 'cross_attention')
        assert hasattr(model, 'classifier')
    
    def test_count_parameters(self, model):
        """Test parameter counting."""
        params = model.count_parameters()
        
        assert 'total' in params
        assert 'trainable' in params
        assert 'frozen' in params
        assert params['trainable'] < params['total']
    
    def test_forward_pass(self, model):
        """Test forward pass with dummy inputs."""
        batch_size = 2
        
        # Create dummy inputs
        pixel_values = torch.randn(batch_size, 3, 224, 224)
        input_ids = torch.randint(0, 49408, (batch_size, 77))
        attention_mask = torch.ones(batch_size, 77)
        
        # Forward pass
        with torch.no_grad():
            output = model(pixel_values, input_ids, attention_mask)
        
        assert output.shape == (batch_size,)
    
    def test_output_range(self, model):
        """Test that sigmoid outputs are in valid range."""
        batch_size = 2
        
        pixel_values = torch.randn(batch_size, 3, 224, 224)
        input_ids = torch.randint(0, 49408, (batch_size, 77))
        attention_mask = torch.ones(batch_size, 77)
        
        with torch.no_grad():
            logits = model(pixel_values, input_ids, attention_mask)
            probs = torch.sigmoid(logits)
        
        assert (probs >= 0).all()
        assert (probs <= 1).all()


class TestFocalLoss:
    """Tests for Focal Loss."""
    
    def test_focal_loss_init(self):
        """Test loss initialization."""
        from losses import FocalLoss
        
        loss_fn = FocalLoss(alpha=0.5, gamma=2.0)
        assert loss_fn.alpha == 0.5
        assert loss_fn.gamma == 2.0
    
    def test_focal_loss_forward(self):
        """Test loss computation."""
        from losses import FocalLoss
        
        loss_fn = FocalLoss(alpha=0.5, gamma=2.0)
        
        logits = torch.randn(8)
        targets = torch.randint(0, 2, (8,)).float()
        
        loss = loss_fn(logits, targets)
        
        assert loss.ndim == 0  # Scalar
        assert loss >= 0
    
    def test_focal_loss_gradient(self):
        """Test that gradients flow."""
        from losses import FocalLoss
        
        loss_fn = FocalLoss()
        
        logits = torch.randn(4, requires_grad=True)
        targets = torch.randint(0, 2, (4,)).float()
        
        loss = loss_fn(logits, targets)
        loss.backward()
        
        assert logits.grad is not None


class TestIntegration:
    """Integration tests."""
    
    @pytest.mark.slow
    def test_training_step(self):
        """Test a single training step."""
        from model import HatefulMemeClassifier
        from losses import FocalLoss
        
        model = HatefulMemeClassifier(hidden_dim=256, num_heads=4)
        loss_fn = FocalLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        # Dummy data
        pixel_values = torch.randn(2, 3, 224, 224)
        input_ids = torch.randint(0, 49408, (2, 77))
        attention_mask = torch.ones(2, 77)
        labels = torch.tensor([0.0, 1.0])
        
        # Training step
        model.train()
        optimizer.zero_grad()
        
        outputs = model(pixel_values, input_ids, attention_mask)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        
        assert loss.item() > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
