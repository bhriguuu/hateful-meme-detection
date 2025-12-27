# Deployment Guide

Complete guide for deploying the Hateful Meme Detection system in production.

## Table of Contents

- [Deployment Options](#deployment-options)
- [Prerequisites](#prerequisites)
- [Local Deployment](#local-deployment)
- [Docker Deployment](#docker-deployment)
- [Cloud Deployment](#cloud-deployment)
- [Scaling Considerations](#scaling-considerations)
- [Monitoring & Logging](#monitoring--logging)
- [Security Best Practices](#security-best-practices)

---

## Deployment Options

| Option | Best For | Cost | Complexity |
|--------|----------|------|------------|
| Local/On-Premise | Development, Small scale | Hardware costs | Low |
| Docker | Consistency, CI/CD | Varies | Medium |
| AWS EC2/ECS | Production, Scaling | ~$50-200/mo | Medium |
| Google Cloud Run | Serverless, Variable load | Pay per use | Low |
| Kubernetes | Large scale, Multi-region | $200+/mo | High |

---

## Prerequisites

### Hardware Requirements

**Minimum (CPU-only):**
- 4 CPU cores
- 8GB RAM
- 10GB storage

**Recommended (GPU):**
- 4+ CPU cores
- 16GB RAM
- NVIDIA GPU with 8GB+ VRAM (RTX 3070, T4, V100)
- 20GB storage

### Software Requirements

- Python 3.8+
- Docker 20.10+ (for containerized deployment)
- NVIDIA Driver 470+ (for GPU)
- CUDA 11.8+ (for GPU)

---

## Local Deployment

### 1. Clone and Setup

```bash
git clone https://github.com/yourusername/hateful-meme-detection.git
cd hateful-meme-detection

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Download Model

```bash
# Option A: Download pre-trained
wget https://your-storage/best_model.pth -O models/best_model.pth

# Option B: Train your own
python src/train.py --data_dir data/hateful_memes --output_dir outputs/
cp outputs/models/best_model.pth models/
```

### 3. Configure Environment

```bash
cp .env.example .env
nano .env  # Edit with your settings
```

### 4. Run Discord Bot

```bash
python discord_bot/bot.py
```

### 5. Run as Service (systemd)

```bash
sudo nano /etc/systemd/system/meme-bot.service
```

```ini
[Unit]
Description=Hateful Meme Detection Bot
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/hateful-meme-detection
Environment=PATH=/opt/hateful-meme-detection/venv/bin
ExecStart=/opt/hateful-meme-detection/venv/bin/python discord_bot/bot.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable meme-bot
sudo systemctl start meme-bot
```

---

## Docker Deployment

### Build Image

```bash
# CPU version
docker build -t hateful-meme:latest .

# GPU version
docker build --target gpu -t hateful-meme:gpu .
```

### Run Container

```bash
# CPU
docker run -d \
  --name meme-bot \
  --restart unless-stopped \
  -e DISCORD_BOT_TOKEN=$DISCORD_BOT_TOKEN \
  -v $(pwd)/models:/app/models:ro \
  hateful-meme:latest

# GPU
docker run -d \
  --name meme-bot \
  --restart unless-stopped \
  --gpus all \
  -e DISCORD_BOT_TOKEN=$DISCORD_BOT_TOKEN \
  -v $(pwd)/models:/app/models:ro \
  hateful-meme:gpu
```

### Docker Compose

```bash
# Start all services
docker-compose up -d discord-bot

# View logs
docker-compose logs -f discord-bot

# Stop
docker-compose down
```

---

## Cloud Deployment

### AWS EC2

#### 1. Launch Instance

- **CPU**: t3.medium or t3.large
- **GPU**: g4dn.xlarge (T4 GPU)
- **AMI**: Ubuntu 22.04 or Deep Learning AMI
- **Storage**: 30GB gp3

#### 2. Security Group

```
Inbound Rules:
- SSH (22) from your IP
- HTTPS (443) if running API

Outbound Rules:
- All traffic (for Discord API)
```

#### 3. Setup Script

```bash
#!/bin/bash
# EC2 setup script

# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker ubuntu

# Clone repository
git clone https://github.com/yourusername/hateful-meme-detection.git
cd hateful-meme-detection

# Setup environment
cp .env.example .env
# Edit .env with your token

# Run
docker-compose up -d discord-bot
```

### Google Cloud Run

```bash
# Build and push
gcloud builds submit --tag gcr.io/PROJECT_ID/hateful-meme

# Deploy
gcloud run deploy hateful-meme \
  --image gcr.io/PROJECT_ID/hateful-meme \
  --platform managed \
  --memory 4Gi \
  --cpu 2 \
  --set-env-vars DISCORD_BOT_TOKEN=$TOKEN
```

### AWS ECS with Fargate

```yaml
# task-definition.json
{
  "family": "hateful-meme-bot",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "4096",
  "containerDefinitions": [
    {
      "name": "bot",
      "image": "YOUR_ECR_IMAGE",
      "essential": true,
      "environment": [
        {"name": "MODEL_PATH", "value": "/app/models/best_model.pth"}
      ],
      "secrets": [
        {"name": "DISCORD_BOT_TOKEN", "valueFrom": "arn:aws:secretsmanager:..."}
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/hateful-meme-bot",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

---

## Scaling Considerations

### Horizontal Scaling

For high-volume Discord servers or multiple servers:

```yaml
# docker-compose.scale.yml
services:
  bot:
    deploy:
      replicas: 3
    environment:
      - SHARD_ID=${SHARD_ID}
      - SHARD_COUNT=3
```

### Load Balancing (API Mode)

```nginx
# nginx.conf
upstream meme_api {
    least_conn;
    server api1:8000;
    server api2:8000;
    server api3:8000;
}

server {
    listen 80;
    location / {
        proxy_pass http://meme_api;
    }
}
```

### Caching

```python
# Add Redis caching for repeated images
import redis
import hashlib

cache = redis.Redis(host='localhost', port=6379)

def predict_with_cache(image_bytes, text):
    key = hashlib.md5(image_bytes + text.encode()).hexdigest()
    
    cached = cache.get(key)
    if cached:
        return json.loads(cached)
    
    result = predictor.predict(image_bytes, text)
    cache.setex(key, 3600, json.dumps(result))  # 1 hour TTL
    
    return result
```

---

## Monitoring & Logging

### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, start_http_server

# Metrics
PREDICTIONS = Counter('predictions_total', 'Total predictions', ['result'])
LATENCY = Histogram('prediction_latency_seconds', 'Prediction latency')

# In prediction function
with LATENCY.time():
    result = predictor.predict(image, text)
    PREDICTIONS.labels(result='hateful' if result['is_hateful'] else 'safe').inc()

# Start metrics server
start_http_server(9090)
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "panels": [
      {
        "title": "Predictions per Minute",
        "targets": [{"expr": "rate(predictions_total[1m])"}]
      },
      {
        "title": "Average Latency",
        "targets": [{"expr": "histogram_quantile(0.95, prediction_latency_seconds)"}]
      }
    ]
  }
}
```

### CloudWatch (AWS)

```python
import boto3

cloudwatch = boto3.client('cloudwatch')

def log_metric(name, value):
    cloudwatch.put_metric_data(
        Namespace='HatefulMemeDetection',
        MetricData=[{
            'MetricName': name,
            'Value': value,
            'Unit': 'Count'
        }]
    )
```

---

## Security Best Practices

### 1. Secrets Management

```bash
# Never commit secrets
echo ".env" >> .gitignore

# Use AWS Secrets Manager or similar
aws secretsmanager create-secret \
  --name discord-bot-token \
  --secret-string $DISCORD_BOT_TOKEN
```

### 2. Network Security

```bash
# Restrict outbound to Discord API only (if possible)
# Use VPC with private subnets
# Enable VPC Flow Logs
```

### 3. Container Security

```dockerfile
# Run as non-root
USER appuser

# Read-only filesystem where possible
# Minimal base image
# Regular security updates
```

### 4. Input Validation

```python
# Validate image size
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB

if len(image_bytes) > MAX_IMAGE_SIZE:
    raise ValueError("Image too large")

# Validate image format
allowed_formats = ['JPEG', 'PNG', 'GIF', 'WEBP']
if image.format not in allowed_formats:
    raise ValueError("Invalid image format")
```

### 5. Rate Limiting

```python
from discord.ext import commands
from collections import defaultdict
import time

rate_limits = defaultdict(list)

@bot.check
async def check_rate_limit(ctx):
    user_id = ctx.author.id
    now = time.time()
    
    # Clean old entries
    rate_limits[user_id] = [t for t in rate_limits[user_id] if now - t < 60]
    
    if len(rate_limits[user_id]) >= 10:  # 10 per minute
        return False
    
    rate_limits[user_id].append(now)
    return True
```

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| OOM Error | Model too large | Use CPU, reduce batch size |
| Slow inference | No GPU | Enable CUDA, use smaller model |
| Bot offline | Token expired | Regenerate Discord token |
| Missing permissions | Bot not configured | Re-invite with correct permissions |

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python discord_bot/bot.py
```

### Health Checks

```bash
# Check if bot is running
docker ps | grep meme-bot

# Check logs
docker logs meme-bot --tail 100

# Test prediction
curl -X POST http://localhost:8000/predict \
  -F "image=@test.jpg" \
  -F "text=test"
```

---

## Maintenance

### Updates

```bash
# Pull latest changes
git pull origin main

# Rebuild container
docker-compose build
docker-compose up -d

# Or with zero downtime
docker-compose up -d --no-deps --build bot
```

### Backup

```bash
# Backup model and configs
tar -czvf backup-$(date +%Y%m%d).tar.gz models/ configs/ .env
```

### Model Updates

```bash
# Download new model
wget https://storage/new_model.pth -O models/best_model_new.pth

# Test new model
python -c "from src.inference import HatefulMemePredictor; p = HatefulMemePredictor('models/best_model_new.pth')"

# Swap models
mv models/best_model.pth models/best_model_old.pth
mv models/best_model_new.pth models/best_model.pth

# Restart bot
docker-compose restart bot
```
