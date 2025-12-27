"""
Training script for Hateful Meme Detection model.

Supports:
- Training from scratch or resuming
- Mixed precision training
- Learning rate scheduling with warmup
- Early stopping
- Comprehensive logging
- Model checkpointing

Usage:
    python src/train.py --data_dir data/hateful_memes --output_dir outputs/
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, 
    recall_score, roc_auc_score
)
from transformers import CLIPProcessor

from model import HatefulMemeClassifier, create_model
from dataset import create_dataloaders, load_jsonl
from losses import FocalLoss, get_loss_function


# ============================================================================
# Training Functions
# ============================================================================

def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
    gradient_clip: float = 1.0,
    use_amp: bool = True
) -> Dict[str, float]:
    """
    Train model for one epoch.
    
    Args:
        model: Model to train
        loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to use
        scaler: Gradient scaler for mixed precision
        gradient_clip: Max gradient norm
        use_amp: Whether to use automatic mixed precision
        
    Returns:
        Dictionary of training metrics
    """
    model.train()
    total_loss = 0
    all_preds = []
    all_probs = []
    all_labels = []
    
    progress = tqdm(loader, desc="Training", leave=False)
    
    for batch in progress:
        # Move to device
        pixel_values = batch['pixel_values'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        # Forward pass with optional mixed precision
        optimizer.zero_grad()
        
        if use_amp and scaler is not None:
            with autocast():
                outputs = model(pixel_values, input_ids, attention_mask)
                loss = criterion(outputs, labels)
            
            # Backward with scaling
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(pixel_values, input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
        
        scheduler.step()
        
        # Track metrics
        total_loss += loss.item()
        probs = torch.sigmoid(outputs).detach()
        preds = (probs > 0.5).float()
        
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        progress.set_postfix({
            'loss': f'{loss.item():.4f}',
            'lr': f'{scheduler.get_last_lr()[0]:.2e}'
        })
    
    # Calculate metrics
    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'f1': f1,
        'auc': auc,
        'lr': scheduler.get_last_lr()[0]
    }


def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[Dict[str, float], list, list, list]:
    """
    Validate model on validation set.
    
    Args:
        model: Model to validate
        loader: Validation data loader
        criterion: Loss function
        device: Device to use
        
    Returns:
        Tuple of (metrics dict, true labels, predictions, probabilities)
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating", leave=False):
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(pixel_values, input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate all metrics
    avg_loss = total_loss / len(loader)
    
    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
    }
    
    try:
        metrics['auc'] = roc_auc_score(all_labels, all_probs)
    except ValueError:
        metrics['auc'] = 0.0
    
    return metrics, all_labels, all_preds, all_probs


# ============================================================================
# Main Training Loop
# ============================================================================

def train(args):
    """Main training function."""
    
    # Setup
    print("=" * 70)
    print("HATEFUL MEME DETECTION - TRAINING")
    print("=" * 70)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Create output directories
    output_dir = Path(args.output_dir)
    model_dir = output_dir / 'models'
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Load processor and create dataloaders
    print("\nLoading data...")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    dataloaders = create_dataloaders(
        data_path=args.data_dir,
        processor=processor,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    
    print(f"Train samples: {len(dataloaders['train_dataset'])}")
    print(f"Val samples: {len(dataloaders['val_dataset'])}")
    
    # Create model
    print("\nCreating model...")
    model_config = {
        'clip_model_name': args.clip_model,
        'hidden_dim': args.hidden_dim,
        'num_heads': args.num_heads,
        'dropout': args.dropout,
        'freeze_clip': True
    }
    
    if args.resume:
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model = create_model(
            config=checkpoint.get('model_config', model_config),
            pretrained_path=None,
            device=device
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        best_f1 = checkpoint.get('best_f1', 0)
        history = checkpoint.get('history', {})
    else:
        model = create_model(config=model_config, device=device)
        start_epoch = 0
        best_f1 = 0
        history = {
            'train_loss': [], 'train_acc': [], 'train_f1': [], 'train_auc': [],
            'val_loss': [], 'val_acc': [], 'val_f1': [], 'val_auc': [],
            'val_precision': [], 'val_recall': [], 'learning_rates': []
        }
    
    # Print model info
    params = model.count_parameters()
    print(f"\nModel Parameters:")
    print(f"  Total: {params['total']:,}")
    print(f"  Trainable: {params['trainable']:,} ({params['trainable_pct']:.2f}%)")
    
    # Calculate class weights for loss function
    train_data = load_jsonl(Path(args.data_dir) / 'train.jsonl')
    n_pos = sum(1 for item in train_data if item.get('label', 0) == 1)
    n_neg = len(train_data) - n_pos
    alpha = n_neg / len(train_data)  # Weight for positive class
    
    # Loss function
    criterion = FocalLoss(alpha=alpha, gamma=args.focal_gamma)
    print(f"\nLoss: Focal Loss (alpha={alpha:.4f}, gamma={args.focal_gamma})")
    
    # Optimizer
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Scheduler
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        total_steps=total_steps,
        pct_start=args.warmup_ratio,
        anneal_strategy='cos',
        div_factor=25,
        final_div_factor=1000
    )
    
    # Mixed precision scaler
    scaler = GradScaler() if args.use_amp and torch.cuda.is_available() else None
    
    # Training config summary
    print(f"\nTraining Configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Warmup ratio: {args.warmup_ratio}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Gradient clip: {args.gradient_clip}")
    print(f"  Mixed precision: {args.use_amp}")
    print(f"  Early stopping: {args.patience} epochs")
    
    # Training loop
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    
    best_model_state = None
    patience_counter = 0
    training_start = time.time()
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        
        print(f"\n{'=' * 60}")
        print(f"EPOCH {epoch + 1}/{args.epochs}")
        print(f"{'=' * 60}")
        
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            device, scaler, args.gradient_clip, args.use_amp
        )
        
        # Validate
        val_metrics, _, _, _ = validate(model, val_loader, criterion, device)
        
        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['train_f1'].append(train_metrics['f1'])
        history['train_auc'].append(train_metrics['auc'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_auc'].append(val_metrics['auc'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['learning_rates'].append(train_metrics['lr'])
        
        # Print metrics
        epoch_time = time.time() - epoch_start
        print(f"\nTraining:")
        print(f"  Loss: {train_metrics['loss']:.4f}")
        print(f"  Acc: {train_metrics['accuracy']:.4f} | F1: {train_metrics['f1']:.4f} | AUC: {train_metrics['auc']:.4f}")
        print(f"\nValidation:")
        print(f"  Loss: {val_metrics['loss']:.4f}")
        print(f"  Acc: {val_metrics['accuracy']:.4f} | F1: {val_metrics['f1']:.4f} | AUC: {val_metrics['auc']:.4f}")
        print(f"  Precision: {val_metrics['precision']:.4f} | Recall: {val_metrics['recall']:.4f}")
        print(f"\nLR: {train_metrics['lr']:.2e} | Time: {epoch_time:.1f}s")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'history': history,
            'best_f1': best_f1,
            'model_config': model_config
        }
        torch.save(checkpoint, model_dir / f'checkpoint_epoch{epoch+1}.pth')
        
        # Check for best model
        current_score = val_metrics['f1']
        if current_score > best_f1:
            best_f1 = current_score
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            
            # Save best model
            torch.save({
                'model_state_dict': best_model_state,
                'model_config': model_config,
                'val_metrics': val_metrics,
                'epoch': epoch + 1
            }, model_dir / 'best_model.pth')
            
            print(f"\n >> NEW BEST MODEL << F1: {best_f1:.4f}")
        else:
            patience_counter += 1
            print(f"\n No improvement ({patience_counter}/{args.patience})")
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Save final model with all metadata
    total_time = time.time() - training_start
    
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': model_config,
        'training_config': {
            'epochs': len(history['train_loss']),
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'loss_function': 'FocalLoss',
            'optimizer': 'AdamW',
            'scheduler': 'OneCycleLR'
        },
        'validation_performance': {
            'val_accuracy': history['val_acc'][-1],
            'val_f1': best_f1,
            'val_auc': history['val_auc'][-1],
            'val_precision': history['val_precision'][-1],
            'val_recall': history['val_recall'][-1]
        },
        'history': history,
        'total_training_time': total_time
    }
    
    torch.save(final_checkpoint, model_dir / 'final_model.pth')
    
    # Save history as JSON
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n{'=' * 70}")
    print("TRAINING COMPLETE")
    print(f"{'=' * 70}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Best F1: {best_f1:.4f}")
    print(f"Models saved to: {model_dir}")


# ============================================================================
# Argument Parser
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train Hateful Meme Detection Model'
    )
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='outputs/',
                        help='Output directory for models and logs')
    
    # Model arguments
    parser.add_argument('--clip_model', type=str, 
                        default='openai/clip-vit-base-patch32',
                        help='CLIP model name')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden dimension for fusion layers')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout probability')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                        help='Warmup ratio for scheduler')
    parser.add_argument('--gradient_clip', type=float, default=1.0,
                        help='Gradient clipping max norm')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal loss gamma parameter')
    parser.add_argument('--patience', type=int, default=3,
                        help='Early stopping patience')
    
    # System arguments
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--use_amp', action='store_true',
                        help='Use automatic mixed precision')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
