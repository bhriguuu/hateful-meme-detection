# Discord Bot - Content Moderation

Real-time hateful meme detection bot for Discord servers using our trained multimodal AI model.

## Features

- 🔍 **Automatic Scanning**: Monitors image uploads in real-time
- 🎯 **Tiered Response System**: Safe → Warning → Auto-delete based on confidence
- 📊 **Statistics Tracking**: View moderation stats with `!mod stats`
- 🔧 **Admin Controls**: Adjustable thresholds for server admins
- ⚡ **Fast Inference**: GPU-accelerated predictions

## Quick Setup

### 1. Create Discord Application

1. Go to [Discord Developer Portal](https://discord.com/developers/applications)
2. Click "New Application" and give it a name
3. Go to "Bot" section and click "Add Bot"
4. **Enable "Message Content Intent"** (required!)
5. Copy the bot token

### 2. Invite Bot to Server

1. Go to "OAuth2" → "URL Generator"
2. Select scopes: `bot`, `applications.commands`
3. Select permissions:
   - Read Messages/View Channels
   - Send Messages
   - Manage Messages (for auto-delete)
   - Add Reactions
   - Read Message History
4. Copy generated URL and open in browser
5. Select your server and authorize

### 3. Configure Environment

```bash
# Create .env file in project root
cp .env.example .env

# Edit with your values
nano .env
```

Required variables:
```env
DISCORD_BOT_TOKEN=your_bot_token_here
MONITORED_CHANNEL_ID=123456789  # Optional: specific channel only
MODEL_PATH=models/best_model.pth
```

### 4. Run the Bot

```bash
# Option 1: Direct
python discord_bot/bot.py

# Option 2: Docker
docker-compose up discord-bot

# Option 3: With GPU
docker-compose up discord-bot-gpu
```

## Commands

| Command | Description | Permission |
|---------|-------------|------------|
| `!mod check` | Analyze the last image in channel | Everyone |
| `!mod stats` | Show moderation statistics | Everyone |
| `!mod info` | Display bot help and info | Everyone |
| `!mod threshold 0.45` | Set classification threshold | Admin |

## Threshold Configuration

The bot uses a tiered response system:

```
Probability Range        Action           Response
─────────────────────────────────────────────────────
< 37%                    ✅ Approve       Checkmark reaction
37% - 42.74%             ⚠️ Warn          Warning reaction + message
> 42.74%                 🛑 Remove        Auto-delete + notification
```

### Adjusting Thresholds

**More Aggressive** (fewer false negatives):
```env
SAFE_THRESHOLD=0.30
OPTIMAL_THRESHOLD=0.40
```

**More Lenient** (fewer false positives):
```env
SAFE_THRESHOLD=0.45
OPTIMAL_THRESHOLD=0.55
```

## Production Deployment

### Using systemd (Linux)

```bash
# Create service file
sudo nano /etc/systemd/system/hateful-meme-bot.service
```

```ini
[Unit]
Description=Hateful Meme Detection Discord Bot
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/hateful-meme-detection
ExecStart=/opt/hateful-meme-detection/venv/bin/python discord_bot/bot.py
Restart=always
RestartSec=10
EnvironmentFile=/opt/hateful-meme-detection/.env

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start
sudo systemctl enable hateful-meme-bot
sudo systemctl start hateful-meme-bot

# Check status
sudo systemctl status hateful-meme-bot

# View logs
journalctl -u hateful-meme-bot -f
```

### Using Docker

```bash
# Build
docker build -t hateful-meme-bot .

# Run
docker run -d \
  --name hateful-meme-bot \
  --restart unless-stopped \
  -e DISCORD_BOT_TOKEN=$DISCORD_BOT_TOKEN \
  -v $(pwd)/models:/app/models:ro \
  hateful-meme-bot
```

### Cloud Deployment Options

- **AWS EC2**: t3.medium for CPU, g4dn.xlarge for GPU
- **Google Cloud**: e2-medium for CPU, n1-standard-4 with T4 GPU
- **DigitalOcean**: Basic Droplet ($12/mo) works for small servers
- **Railway/Render**: Easy deployment from GitHub

## Monitoring

### Health Check Endpoint

For production monitoring, the bot logs:
- Total images checked
- Classification breakdown
- Error count
- Response times

### Recommended Monitoring

```python
# Add to bot.py for external monitoring
@bot.command(name='health')
async def health_check(ctx):
    """Health check for monitoring systems."""
    await ctx.send(json.dumps({
        'status': 'healthy',
        'checked': stats['total_checked'],
        'uptime': time.time() - start_time
    }))
```

## Troubleshooting

### Bot Not Responding

1. Check "Message Content Intent" is enabled in Developer Portal
2. Verify bot has correct permissions in channel
3. Check logs for errors: `docker logs hateful-meme-bot`

### High Memory Usage

1. Ensure model is loaded once at startup
2. Use CPU inference for lower memory
3. Increase swap space if needed

### Slow Predictions

1. Enable GPU if available
2. Check batch size settings
3. Consider model quantization

## Security Considerations

⚠️ **Important:**
- Never commit `.env` or bot tokens to git
- Use environment variables for all secrets
- Rotate bot token if compromised
- Implement rate limiting for commands
- Log moderation actions for audit trails

## Support

For issues:
1. Check bot logs first
2. Review [Troubleshooting](#troubleshooting)
3. Open GitHub issue with logs and configuration
