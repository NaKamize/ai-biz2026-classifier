# AI-Biz2026: Fashion-MNIST Classification

This repository contains my solution for the computer vision competition organized within:

**AI-Biz2026**  
International Workshop: Artificial Intelligence of and for Business (AI-Biz2026)  
associated with JSAI International Symposia on AI 2026 (JSAI-isAI 2026)

## Final Result

**3rd place out of 38 teams**

| | Score |
|---|---|
| Public score | 0.95440 |
| Private score | 0.95560 |

Top leaderboard:
1. KaiwenLin / 114ZU1124 — 0.96340
2. Matthew — 0.95700
3. Jozef Makis — **0.95560**

## Task

Classify Fashion-MNIST images into 10 clothing categories. Data is provided in CSV format (label + 784 pixel values per row). The test set has no labels and predictions are evaluated on accuracy.

## Solution Overview

The final pipeline is a multi-seed ResNet with Squeeze-Excitation blocks, a specialist sub-model for the most confused class cluster, and test-time augmentation. All components were chosen specifically to address the hardest part of this dataset: distinguishing T-shirt, Pullover, Coat, and Shirt from each other.

### Architecture

Custom ResNet with Squeeze-Excitation (SE) blocks and Swish activations, trained natively on grayscale 28x28 input. Three stages of 64, 128, and 256 filters with residual connections and SE recalibration after each block. SE blocks let the network learn which feature channels matter most per sample, which helps separate visually similar garments. Swish activations were chosen over ReLU for smoother gradient flow.

### Base Ensemble

Five base models were trained with different random seeds (11, 23, 47, 59, 73). Each was trained independently on a 90/10 stratified split and their softmax probability outputs were averaged. Multi-seed ensembling reduces variance and consistently improves accuracy compared to a single model, even with the same architecture.

### Learning Rate Schedule

AdamW optimizer (lr=2e-3, weight decay=1e-4) with ReduceLROnPlateau (factor=0.5, patience=2, min lr=1e-5) and EarlyStopping (patience=5, restore best weights). Halving the learning rate on plateau avoids overshooting fine-grained decision boundaries late in training.

### Data Augmentation

Random translation, random zoom, random contrast, and random cutout (8x8 patch). Cutout forces the model to use global texture rather than relying on a single discriminative region, which is useful for garments that share texture but differ in shape. Augmentation was applied on the fly during training using a tf.data pipeline.

### Specialist Model for the Shirt Cluster

A separate 4-class model was trained exclusively on T-shirt (0), Pullover (2), Coat (4), and Shirt (6). These four classes had the lowest per-class accuracy and the highest mutual confusion. Training a dedicated model on only these four classes lets it develop sharper decision boundaries for collar shape, sleeve style, and torso cut that the 10-class model cannot learn as precisely. Three seeds were used and averaged.

### Routing and Fusion

At inference time, samples predicted by the base ensemble into the shirt cluster AND with low margin between the top two predictions are routed to the specialist. The specialist probabilities are blended into the base probabilities for those samples. The routing threshold (margin) and blend weight were selected by grid search on the validation set. Routing only low-confidence shirt-cluster predictions avoids disturbing high-confidence correct predictions from the base model.

### Test-Time Augmentation

Five forward passes per model: one clean pass and four passes with random augmentation. Probabilities were averaged across passes.

## Repository Contents

- [ai-biz-train-final-submission.ipynb](ai-biz-train-final-submission.ipynb) — final submitted notebook

## What Did Not Work

| Approach | Outcome |
|---|---|
| CBAM spatial attention | Worse than SE only |
| Random rotation | Slight degradation |
| More TTA passes (N=8) | No improvement |
| EfficientNet transfer learning | Did not outperform native ResNet |
| MixUp | Did not improve |
| Class weight boosting for Shirt | Added noise |
| WideResNet 2x width | Worse |

