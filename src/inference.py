"""
Inference module for Hateful Meme Detection.

Provides easy-to-use prediction interfaces for:
- Single image inference
- Batch inference
- Streaming inference

Optimized for production deployment with:
- Model caching
- GPU acceleration
- Configurable thresholds
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor

try:
    import albumentations as A
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False

from .model import HatefulMemeClassifier


class HatefulMemePredictor:
    """
    Production-ready predictor for hateful meme detection.
    
    Provides simple interface for making predictions on images with text.
    Handles all preprocessing internally.
    
    Args:
        model_path: Path to trained model checkpoint
        device: Device to run inference on ('cuda' or 'cpu')
        threshold: Classification threshold (default: 0.4274 from ROC analysis)
    
    Example:
        >>> predictor = HatefulMemePredictor("models/best_model.pth")
        >>> result = predictor.predict("meme.jpg", "text on meme")
        >>> print(f"Hateful: {result['is_hateful']}, Confidence: {result['confidence']:.2%}")
    """
    
    def __init__(
        self,
        model_path: Union[str, Path],
        device: str = 'cuda',
        threshold: float = 0.4274
    ):
        self.device = torch.device(
            device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        )
        self.threshold = threshold
        
        # Load model
        self.model, self.config = self._load_model(model_path)
        self.model.eval()
        
        # Load processor
        self.processor = CLIPProcessor.from_pretrained(
            self.config.get('clip_model_name', 'openai/clip-vit-base-patch32')
        )
        
        # Setup transform
        self.transform = self._get_transform()
        
        print(f"Loaded model on {self.device}")
        print(f"Classification threshold: {self.threshold}")
    
    def _load_model(self, model_path: Union[str, Path]):
        """Load model from checkpoint."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Extract config
        config = checkpoint.get('model_config', {
            'clip_model_name': 'openai/clip-vit-base-patch32',
            'hidden_dim': 512,
            'num_heads': 8,
            'dropout': 0.3,
            'freeze_clip': True
        })
        
        # Create model
        model = HatefulMemeClassifier(**config)
        
        # Load weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        
        # Update threshold if available
        if 'inference' in checkpoint:
            self.threshold = checkpoint['inference'].get(
                'optimal_threshold', self.threshold
            )
        
        return model, config
    
    def _get_transform(self):
        """Get image transformation pipeline."""
        if ALBUMENTATIONS_AVAILABLE:
            return A.Compose([
                A.Resize(224, 224),
                A.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711]
                ),
            ])
        return None
    
    def _preprocess_image(self, image: Union[str, Path, Image.Image, np.ndarray]) -> torch.Tensor:
        """Preprocess image for model input."""
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, Image.Image):
            image = image.convert('RGB')
        
        # Convert to numpy
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image
        
        # Apply transform
        if self.transform:
            augmented = self.transform(image=image_np)
            image_np = augmented['image']
        else:
            # Basic resize without albumentations
            image = Image.fromarray(image_np).resize((224, 224))
            image_np = np.array(image) / 255.0
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float()
        
        return image_tensor.unsqueeze(0).to(self.device)
    
    def _preprocess_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Preprocess text for model input."""
        if not text or not isinstance(text, str):
            text = "no text"
        
        text = text.strip()[:512]  # Truncate long text
        
        inputs = self.processor(
            text=[text],
            return_tensors="pt",
            padding='max_length',
            truncation=True,
            max_length=77
        )
        
        return {
            'input_ids': inputs['input_ids'].to(self.device),
            'attention_mask': inputs['attention_mask'].to(self.device)
        }
    
    def predict(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        text: str,
        return_logits: bool = False
    ) -> Dict:
        """
        Predict if a meme is hateful.
        
        Args:
            image: Image path, PIL Image, or numpy array
            text: Text content of the meme
            return_logits: Whether to include raw logits in output
            
        Returns:
            Dictionary with prediction results:
                - is_hateful: Boolean classification
                - label: "HATEFUL" or "NOT HATEFUL"
                - probability: Probability of being hateful
                - confidence: Confidence score (max of prob, 1-prob)
                - threshold: Classification threshold used
        """
        # Preprocess
        pixel_values = self._preprocess_image(image)
        text_inputs = self._preprocess_text(text)
        
        # Predict
        with torch.no_grad():
            logits = self.model(
                pixel_values,
                text_inputs['input_ids'],
                text_inputs['attention_mask']
            )
            probability = torch.sigmoid(logits).item()
        
        # Build result
        is_hateful = probability > self.threshold
        confidence = max(probability, 1 - probability)
        
        result = {
            'is_hateful': is_hateful,
            'label': 'HATEFUL' if is_hateful else 'NOT HATEFUL',
            'probability': probability,
            'confidence': confidence,
            'threshold': self.threshold
        }
        
        if return_logits:
            result['logits'] = logits.item()
        
        return result
    
    def predict_batch(
        self,
        images: List[Union[str, Path, Image.Image]],
        texts: List[str],
        batch_size: int = 16
    ) -> List[Dict]:
        """
        Batch prediction for multiple memes.
        
        Args:
            images: List of image paths or PIL Images
            texts: List of text strings
            batch_size: Batch size for processing
            
        Returns:
            List of prediction dictionaries
        """
        if len(images) != len(texts):
            raise ValueError("Number of images and texts must match")
        
        results = []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            batch_texts = texts[i:i + batch_size]
            
            # Process batch
            pixel_values = torch.cat([
                self._preprocess_image(img) for img in batch_images
            ], dim=0)
            
            # Process texts
            text_inputs = self.processor(
                text=batch_texts,
                return_tensors="pt",
                padding='max_length',
                truncation=True,
                max_length=77
            )
            
            input_ids = text_inputs['input_ids'].to(self.device)
            attention_mask = text_inputs['attention_mask'].to(self.device)
            
            # Predict
            with torch.no_grad():
                logits = self.model(pixel_values, input_ids, attention_mask)
                probabilities = torch.sigmoid(logits)
            
            # Build results
            for prob in probabilities:
                prob_val = prob.item()
                is_hateful = prob_val > self.threshold
                
                results.append({
                    'is_hateful': is_hateful,
                    'label': 'HATEFUL' if is_hateful else 'NOT HATEFUL',
                    'probability': prob_val,
                    'confidence': max(prob_val, 1 - prob_val),
                    'threshold': self.threshold
                })
        
        return results
    
    def get_detailed_analysis(
        self,
        image: Union[str, Path, Image.Image],
        text: str
    ) -> Dict:
        """
        Get detailed analysis including feature insights.
        
        Args:
            image: Image input
            text: Text content
            
        Returns:
            Detailed analysis dictionary
        """
        prediction = self.predict(image, text, return_logits=True)
        
        # Add classification zone
        prob = prediction['probability']
        if prob < 0.37:
            zone = 'SAFE'
            zone_color = 'green'
        elif prob < self.threshold:
            zone = 'WARNING'
            zone_color = 'yellow'
        else:
            zone = 'HARMFUL'
            zone_color = 'red'
        
        prediction.update({
            'zone': zone,
            'zone_color': zone_color,
            'text_analyzed': text[:100] + '...' if len(text) > 100 else text,
            'model_info': {
                'architecture': 'CLIP + Cross-Attention Fusion',
                'threshold': self.threshold,
                'device': str(self.device)
            }
        })
        
        return prediction
    
    def set_threshold(self, threshold: float):
        """Update classification threshold."""
        self.threshold = threshold
        print(f"Threshold updated to: {threshold}")


def create_deployment_package(
    model_path: Union[str, Path],
    output_path: Union[str, Path],
    include_processor: bool = True
):
    """
    Create a deployment package with model and configuration.
    
    Args:
        model_path: Path to trained model
        output_path: Output path for deployment package
        include_processor: Whether to include processor config
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Create deployment checkpoint
    deployment = {
        'model_state_dict': checkpoint.get('model_state_dict', checkpoint),
        'model_config': checkpoint.get('model_config', {
            'clip_model_name': 'openai/clip-vit-base-patch32',
            'hidden_dim': 512,
            'num_heads': 8,
            'dropout': 0.3,
            'freeze_clip': True
        }),
        'inference': {
            'optimal_threshold': 0.4274,
            'input_size': 224,
            'max_text_length': 77
        }
    }
    
    # Add validation metrics if available
    if 'validation_performance' in checkpoint:
        deployment['validation_performance'] = checkpoint['validation_performance']
    
    # Save deployment package
    torch.save(deployment, output_path / 'deployment_package.pth')
    
    # Save inference config as JSON
    inference_config = {
        'model_file': 'deployment_package.pth',
        'clip_model': deployment['model_config'].get(
            'clip_model_name', 'openai/clip-vit-base-patch32'
        ),
        'threshold': deployment['inference']['optimal_threshold'],
        'input_size': 224,
        'normalization': {
            'mean': [0.48145466, 0.4578275, 0.40821073],
            'std': [0.26862954, 0.26130258, 0.27577711]
        }
    }
    
    with open(output_path / 'inference_config.json', 'w') as f:
        json.dump(inference_config, f, indent=2)
    
    print(f"Deployment package created at: {output_path}")


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to image')
    parser.add_argument('--text', type=str, required=True,
                        help='Text content')
    parser.add_argument('--threshold', type=float, default=0.4274,
                        help='Classification threshold')
    
    args = parser.parse_args()
    
    predictor = HatefulMemePredictor(
        model_path=args.model,
        threshold=args.threshold
    )
    
    result = predictor.get_detailed_analysis(args.image, args.text)
    
    print("\nPrediction Results")
    print("=" * 50)
    print(f"Label: {result['label']}")
    print(f"Probability: {result['probability']:.4f}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Zone: {result['zone']}")
