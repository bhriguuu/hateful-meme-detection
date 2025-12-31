"""
Discord Bot for Real-Time Hateful Meme Detection

Production-ready content moderation bot using the trained multimodal classifier.
Features tiered response system based on confidence levels.

Usage:
    export DISCORD_BOT_TOKEN="your_token"
    export MONITORED_CHANNEL_ID="channel_id"
    python discord_bot/bot.py

Environment Variables:
    DISCORD_BOT_TOKEN: Bot authentication token
    MONITORED_CHANNEL_ID: Channel to monitor (optional, monitors all if not set)
    MODEL_PATH: Path to model checkpoint (default: models/best_model.pth)
"""

import os
import sys
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Optional

import discord
from discord.ext import commands
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Bot configuration from environment variables."""
    
    # Discord settings
    BOT_TOKEN = os.getenv('DISCORD_BOT_TOKEN', '')
    MONITORED_CHANNEL_ID = os.getenv('MONITORED_CHANNEL_ID', None)
    COMMAND_PREFIX = os.getenv('COMMAND_PREFIX', '!mod ')
    
    # Model settings
    MODEL_PATH = os.getenv('MODEL_PATH', 'models/best_model.pth')
    
    # Thresholds (from ROC analysis)
    OPTIMAL_THRESHOLD = float(os.getenv('OPTIMAL_THRESHOLD', '0.4274'))
    SAFE_THRESHOLD = float(os.getenv('SAFE_THRESHOLD', '0.37'))
    WARNING_THRESHOLD = float(os.getenv('WARNING_THRESHOLD', '0.42'))
    
    # Validate
    @classmethod
    def validate(cls):
        if not cls.BOT_TOKEN:
            raise ValueError("DISCORD_BOT_TOKEN environment variable not set")
        if cls.MONITORED_CHANNEL_ID:
            cls.MONITORED_CHANNEL_ID = int(cls.MONITORED_CHANNEL_ID)


# ============================================================================
# MODEL ARCHITECTURE (Matching training code)
# ============================================================================

class CrossAttentionFusion(nn.Module):
    """Cross-attention mechanism for image-text fusion."""
    
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.img_to_txt_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.txt_to_img_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(embed_dim)
    
    def forward(self, img_feat, txt_feat):
        img_feat = img_feat.unsqueeze(1)
        txt_feat = txt_feat.unsqueeze(1)
        
        img_attended, _ = self.img_to_txt_attn(img_feat, txt_feat, txt_feat)
        img_feat = self.norm1(img_feat + img_attended)
        
        txt_attended, _ = self.txt_to_img_attn(txt_feat, img_feat, img_feat)
        txt_feat = self.norm2(txt_feat + txt_attended)
        
        combined = img_feat + txt_feat
        combined = self.norm3(combined + self.ffn(combined))
        
        return combined.squeeze(1)


class HatefulMemeClassifier(nn.Module):
    """Multimodal classifier for hateful meme detection."""
    
    def __init__(
        self,
        clip_model_name="openai/clip-vit-base-patch32",
        hidden_dim=512,
        num_heads=8,
        dropout=0.3,
        freeze_clip=True
    ):
        super().__init__()
        
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        self.embed_dim = self.clip.config.projection_dim
        
        if freeze_clip:
            for param in self.clip.parameters():
                param.requires_grad = False
        
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
        
        self.cross_attention = CrossAttentionFusion(hidden_dim, num_heads, dropout)
        
        self.concat_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
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
    
    def forward(self, pixel_values, input_ids, attention_mask):
        image_features = self.clip.get_image_features(pixel_values=pixel_values)
        text_features = self.clip.get_text_features(
            input_ids=input_ids, attention_mask=attention_mask
        )
        
        image_features = F.normalize(image_features, p=2, dim=-1)
        text_features = F.normalize(text_features, p=2, dim=-1)
        
        img_proj = self.img_proj(image_features)
        txt_proj = self.txt_proj(text_features)
        
        cross_attn_out = self.cross_attention(img_proj, txt_proj)
        concat_out = self.concat_proj(torch.cat([img_proj, txt_proj], dim=-1))
        
        combined = torch.cat([cross_attn_out, concat_out], dim=-1)
        logits = self.classifier(combined)
        
        return logits.squeeze(-1)


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model(model_path: str, device: torch.device):
    """Load trained model from checkpoint."""
    print(f"Loading model from {model_path}...")
    
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        model_config = checkpoint.get('model_config', {
            'clip_model_name': 'openai/clip-vit-base-patch32',
            'hidden_dim': 512,
            'num_heads': 8,
            'dropout': 0.3,
            'freeze_clip': True
        }) if isinstance(checkpoint, dict) and 'model_config' in checkpoint else {
            'clip_model_name': 'openai/clip-vit-base-patch32',
            'hidden_dim': 512,
            'num_heads': 8,
            'dropout': 0.3,
            'freeze_clip': True
        }
        
        model = HatefulMemeClassifier(**model_config)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(device)
        model.eval()
        
        print(f"✅ Model loaded successfully")
        print(f"   Threshold: {Config.OPTIMAL_THRESHOLD:.4f}")
        
        return model
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise


# ============================================================================
# BOT SETUP
# ============================================================================

# Initialize
Config.validate()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = load_model(Config.MODEL_PATH, device)

# Bot intents
intents = discord.Intents.default()
intents.message_content = True
intents.messages = True

bot = commands.Bot(command_prefix=Config.COMMAND_PREFIX, intents=intents)

# Statistics tracking
stats = {
    'total_checked': 0,
    'safe': 0,
    'warnings': 0,
    'deleted': 0,
    'errors': 0
}


# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def predict_content(image: Image.Image, text: str = "") -> float:
    """
    Predict hateful probability for image-text pair.
    
    Args:
        image: PIL Image
        text: Associated text
        
    Returns:
        Probability of being hateful (0-1)
    """
    try:
        # Prepare inputs
        inputs = processor(
            text=[text if text else "a meme"],
            images=image,
            return_tensors="pt",
            padding=True
        )
        
        pixel_values = inputs['pixel_values'].to(device)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        
        # Predict
        with torch.no_grad():
            logits = model(pixel_values, input_ids, attention_mask)
            probability = torch.sigmoid(logits).item()
        
        return probability
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return 0.0


# ============================================================================
# BOT EVENTS
# ============================================================================

@bot.event
async def on_ready():
    """Called when bot is ready."""
    print(f"\n{'=' * 60}")
    print(f"🤖 CONTENT MODERATION BOT ONLINE")
    print(f"{'=' * 60}")
    print(f"Bot: {bot.user.name} (ID: {bot.user.id})")
    print(f"Device: {device}")
    print(f"Monitoring: {'All channels' if not Config.MONITORED_CHANNEL_ID else f'Channel {Config.MONITORED_CHANNEL_ID}'}")
    print(f"\nThresholds:")
    print(f"  ✅ Safe: < {Config.SAFE_THRESHOLD:.0%}")
    print(f"  ⚠️  Warning: {Config.SAFE_THRESHOLD:.0%} - {Config.OPTIMAL_THRESHOLD:.0%}")
    print(f"  🛑 Delete: > {Config.OPTIMAL_THRESHOLD:.0%}")
    print(f"{'=' * 60}\n")


@bot.event
async def on_message(message: discord.Message):
    """Process incoming messages for image content."""
    
    # Ignore bot messages
    if message.author == bot.user:
        return
    
    # Check channel restriction
    if Config.MONITORED_CHANNEL_ID and message.channel.id != Config.MONITORED_CHANNEL_ID:
        await bot.process_commands(message)
        return
    
    # Process commands first
    await bot.process_commands(message)
    
    # Check for image attachments
    if not message.attachments:
        return
    
    for attachment in message.attachments:
        # Check if image
        if not any(attachment.filename.lower().endswith(ext) 
                   for ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']):
            continue
        
        print(f"\n📷 Processing: {attachment.filename}")
        print(f"   User: {message.author.name}")
        print(f"   Channel: {message.channel.name}")
        
        try:
            # Download image
            response = requests.get(attachment.url, timeout=10)
            image = Image.open(BytesIO(response.content)).convert('RGB')
            
            # Get text content
            text_content = message.content if message.content else ""
            
            # Predict
            probability = predict_content(image, text_content)
            stats['total_checked'] += 1
            
            print(f"   Probability: {probability:.2%}")
            
            # Take action based on probability
            if probability < Config.SAFE_THRESHOLD:
                # Safe content
                await message.add_reaction('✅')
                stats['safe'] += 1
                print(f"   ✅ SAFE")
                
            elif probability < Config.OPTIMAL_THRESHOLD:
                # Warning zone
                await message.add_reaction('⚠️')
                await message.reply(
                    f"⚠️ **Content Advisory**\n"
                    f"This content has been flagged for review.\n"
                    f"Confidence: {probability:.1%}",
                    mention_author=False
                )
                stats['warnings'] += 1
                print(f"   ⚠️ WARNING")
                
            else:
                # Harmful content - delete
                await message.delete()
                await message.channel.send(
                    f"🛑 **Content Removed**\n"
                    f"A message from {message.author.mention} was removed "
                    f"for violating community guidelines.\n"
                    f"Confidence: {probability:.1%}\n\n"
                    f"*If you believe this was an error, please contact a moderator.*"
                )
                stats['deleted'] += 1
                print(f"   🛑 DELETED")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            stats['errors'] += 1
            await message.add_reaction('❓')


# ============================================================================
# BOT COMMANDS
# ============================================================================

@bot.command(name='check')
async def manual_check(ctx: commands.Context):
    """Manually check the last image in the channel."""
    async for message in ctx.channel.history(limit=50):
        if not message.attachments:
            continue
            
        for attachment in message.attachments:
            if not any(attachment.filename.lower().endswith(ext) 
                       for ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']):
                continue
            
            try:
                response = requests.get(attachment.url, timeout=10)
                image = Image.open(BytesIO(response.content)).convert('RGB')
                text_content = message.content if message.content else ""
                
                probability = predict_content(image, text_content)
                
                # Determine status
                if probability < Config.SAFE_THRESHOLD:
                    status = "✅ Safe"
                    color = discord.Color.green()
                elif probability < Config.OPTIMAL_THRESHOLD:
                    status = "⚠️ Suspicious"
                    color = discord.Color.orange()
                else:
                    status = "🛑 Harmful"
                    color = discord.Color.red()
                
                embed = discord.Embed(
                    title="Content Analysis",
                    color=color,
                    timestamp=datetime.now()
                )
                embed.add_field(name="Status", value=status, inline=True)
                embed.add_field(name="Confidence", value=f"{probability:.1%}", inline=True)
                embed.add_field(
                    name="Message", 
                    value=f"[Jump to message]({message.jump_url})", 
                    inline=False
                )
                embed.set_thumbnail(url=attachment.url)
                
                await ctx.send(embed=embed)
                return
                
            except Exception as e:
                await ctx.send(f"❌ Error analyzing image: {e}")
                return
    
    await ctx.send("No images found in recent messages.")


@bot.command(name='stats')
async def show_stats(ctx: commands.Context):
    """Show moderation statistics."""
    embed = discord.Embed(
        title="📊 Moderation Statistics",
        color=discord.Color.blue(),
        timestamp=datetime.now()
    )
    
    embed.add_field(name="Total Checked", value=stats['total_checked'], inline=True)
    embed.add_field(name="✅ Safe", value=stats['safe'], inline=True)
    embed.add_field(name="⚠️ Warnings", value=stats['warnings'], inline=True)
    embed.add_field(name="🛑 Deleted", value=stats['deleted'], inline=True)
    embed.add_field(name="❌ Errors", value=stats['errors'], inline=True)
    
    if stats['total_checked'] > 0:
        safe_rate = stats['safe'] / stats['total_checked'] * 100
        embed.add_field(name="Safe Rate", value=f"{safe_rate:.1f}%", inline=True)
    
    embed.set_footer(text=f"Running on {device}")
    
    await ctx.send(embed=embed)


@bot.command(name='info')
async def show_info(ctx: commands.Context):
    """Display bot information."""
    embed = discord.Embed(
        title="🛡️ Content Moderation Bot",
        description="AI-powered multimodal content moderation",
        color=discord.Color.purple()
    )
    
    embed.add_field(
        name="📝 Commands",
        value=(
            f"`{Config.COMMAND_PREFIX}check` - Analyze last image\n"
            f"`{Config.COMMAND_PREFIX}stats` - View statistics\n"
            f"`{Config.COMMAND_PREFIX}info` - This message"
        ),
        inline=False
    )
    
    embed.add_field(
        name="🎯 Thresholds",
        value=(
            f"✅ **Safe**: < {Config.SAFE_THRESHOLD:.0%}\n"
            f"⚠️ **Warning**: {Config.SAFE_THRESHOLD:.0%} - {Config.OPTIMAL_THRESHOLD:.0%}\n"
            f"🛑 **Remove**: > {Config.OPTIMAL_THRESHOLD:.0%}"
        ),
        inline=False
    )
    
    embed.add_field(
        name="🧠 Model",
        value=(
            "• Architecture: CLIP + Cross-Attention\n"
            "• Training: Facebook Hateful Memes\n"
            "• Parameters: 157M (5.9M trainable)"
        ),
        inline=False
    )
    
    embed.set_footer(text="Multimodal Hateful Meme Detection System")
    
    await ctx.send(embed=embed)


@bot.command(name='threshold')
@commands.has_permissions(administrator=True)
async def set_threshold(ctx: commands.Context, value: float):
    """Set the classification threshold (admin only)."""
    if not 0 < value < 1:
        await ctx.send("❌ Threshold must be between 0 and 1")
        return
    
    Config.OPTIMAL_THRESHOLD = value
    await ctx.send(f"✅ Threshold updated to {value:.2%}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("Starting Content Moderation Bot...")
    print("=" * 60 + "\n")
    
    bot.run(Config.BOT_TOKEN)
