# Accuracy Improvement Plan: 61% → as high as this benchmark allows

## Where the 61% comes from

Current validation numbers (from `configs/inference_config.json`):
accuracy **0.61**, AUC **0.676**, recall **0.476** on the balanced
500-sample Hateful Memes dev set.

Ranked causes, most impactful first:

1. **Fully frozen CLIP ViT-B/32 backbone.** Only 3.76% of parameters
   trained. CLIP's pooled embeddings were trained for image–text
   *matching*, not for detecting when a benign image plus benign text
   combine into hate. The dataset was explicitly built with "benign
   confounders" that defeat exactly this kind of shallow fusion.
   Frozen CLIP-B/32 + MLP head plateaus at ~60–65% accuracy — the model
   was performing as expected for its architecture, not misbehaving.
2. **Degenerate cross-attention.** The fusion module attended over
   sequences of length 1 (single pooled vectors). Softmax over one key
   always returns weight 1.0, so the "cross-attention" reduced to a
   linear layer. There was no real token-level image–text interaction.
3. **Harmful augmentations.** `HorizontalFlip` mirrors the text baked
   into meme images (CLIP reads that text); heavy hue/noise/blur pushes
   inputs away from CLIP's pretraining distribution.
4. **Model selection & threshold mismatch.** Best checkpoint was chosen
   by F1 at a hard-coded 0.5 threshold while inference used 0.4274;
   recall of 0.476 means nearly half of hateful memes were missed.
5. **Silent data corruption paths.** Without albumentations, images were
   fed un-resized and un-normalized (0–255); failed image loads became
   black placeholders with real labels.

## What changed in this branch

| File | Change |
|------|--------|
| `src/model.py` | Token-level cross-attention (`use_token_fusion`): patch tokens attend to text tokens with padding masks — real multimodal interaction. Partial fine-tuning (`unfreeze_layers`): top-N blocks of both CLIP encoders unfrozen. Backward compatible — old checkpoints still load with default flags. |
| `src/train.py` | `--unfreeze_layers` (default 4) + `--backbone_lr` (default 1e-5) with discriminative LR param groups; best model selected by val AUC; decision threshold tuned on validation and saved into the checkpoint (`inference.optimal_threshold`, which `src/inference.py` already reads); seeding for reproducibility. |
| `src/dataset.py` | Removed horizontal flip and heavy noise/blur/hue augmentation; fixed the no-albumentations fallback (resize + CLIP normalization). |
| `src/inference.py` | Fallback preprocessing now applies CLIP normalization (was `/255` only — a train/serve mismatch). |

## How to retrain (in order of expected gain)

```bash
# Step 1 — new architecture, fine-tuned ViT-B/32 (expect ~68–72% acc)
python src/train.py --data_dir data/hateful_memes --output_dir outputs/run1 \
    --unfreeze_layers 4 --backbone_lr 1e-5 --learning_rate 2e-4 \
    --epochs 15 --patience 4 --use_amp

# Step 2 — upgrade backbone to ViT-L/14 (expect ~72–78% acc, needs ~16GB GPU;
# reduce --batch_size to 16 or 8 if you hit OOM)
python src/train.py --data_dir data/hateful_memes --output_dir outputs/run2 \
    --clip_model openai/clip-vit-large-patch14 \
    --unfreeze_layers 2 --backbone_lr 5e-6 --learning_rate 1e-4 \
    --batch_size 16 --epochs 15 --patience 4 --use_amp

# Step 3 — ensemble: train 3 seeds and average sigmoid probabilities
for seed in 42 43 44; do
    python src/train.py --data_dir data/hateful_memes \
        --output_dir outputs/seed$seed --seed $seed \
        --clip_model openai/clip-vit-large-patch14 \
        --unfreeze_layers 2 --backbone_lr 5e-6 --learning_rate 1e-4 \
        --batch_size 16 --epochs 15 --use_amp
done
```

Further gains after that, in order:
- **Caption enrichment:** generate a BLIP-2/LLaVA caption per image and
  append it to the meme text before tokenization (helps CLIP's 77-token
  text encoder see what's *in* the image, e.g. "a goat" next to slur text).
- **External VLM route:** zero-/few-shot classification with a large
  vision-language model API scores competitively with no training at all,
  and an ensemble of your classifier + a VLM judge is the strongest
  realistic system.

## Honest calibration on the 90% target

On the **official Hateful Memes benchmark**, published state of the art is
roughly **78–80% accuracy / ~0.85–0.87 AUROC** (challenge winner: 0.845
AUROC; trained human annotators: ~85% accuracy). **No published system
reaches 90% accuracy on this dev/test set.** The changes here should move
you from 61% into the **mid-to-high 70s**, which is competitive research
territory. If the 90% figure is a hard requirement, it must come from one
of: (a) reporting AUROC instead of accuracy, (b) evaluating on an
easier/custom curated test set, or (c) an LLM-with-vision judge pipeline
on a narrower definition of "hateful". Setting that expectation now avoids
chasing a number the benchmark itself does not permit.
