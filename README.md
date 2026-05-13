# AI-Biz2026: Fashion-MNIST Classification

This repository is my working project for the competition organized within:

**AI-Biz2026**  
International Workshop: Artificial Intelligence of and for Business (AI-Biz2026)  
associated with JSAI International Symposia on AI 2026 (JSAI-isAI 2026)

## Current Status

I am currently in **3rd place** on the public leaderboard.

- Team: **Jozef Makis**
- Score: **0.94220**
- Last submission: **2026-05-13 20:06:59**
- Submission count: **2**

Top leaderboard snapshot (at export time):
1. KaiwenLin/ 114ZU1124 - 0.96340
2. Matthew - 0.95700
3. Jozef Makis - 0.94220

## Project Scope

The task is to classify Fashion-MNIST images into 10 classes. The data is provided in CSV format:

- train: label + 784 pixels
- test: 784 pixels

The main goal is to get the best possible leaderboard score while keeping a pipeline that is fast to iterate and rerun.

## How I Trained the Model

I tested multiple approaches. I started with transfer learning and then moved to a simpler grayscale-native model, which was more stable for this task.

Training pipeline:
- pixel normalization to the [0, 1] range
- 90/10 stratified train/validation split
- ResNet-style CNN for 28x28 grayscale input
- multi-seed ensemble (probability averaging)
- additional specialist model for the most frequently confused classes
- fusion of base + specialist predictions before final class selection

Training notebook (Colab):
https://colab.research.google.com/drive/1r6s5XDZGa_L8_YHZZ0MZgjth9-kxHdvX?usp=drive_link

## Repository Contents

- `ai-biz-2026-spring-task-3/`
	- `fashion-mnist_train.csv`
	- `fashion-mnist_test.csv`
	- `fashion-mnist_test_sample_submission.csv`
- `fashion_mnist_transfer_learning_efficientnet (4).ipynb` - older baseline
- `fashion_mnist_resnet_specialist_pipeline.ipynb` - current main pipeline
- `kaggle_submit_from_npy.ipynb` - submission export from saved probabilities

## How to Run

1. Open `fashion_mnist_resnet_specialist_pipeline.ipynb`.
2. Check data paths (`DATA_DIR`) for your environment (local/Colab).
3. Run training and validation.
4. Generate `test_probs_resnet_specialist.npy`.
5. Open `kaggle_submit_from_npy.ipynb` and create `submission.csv`.

## Note

I keep updating this repo based on leaderboard feedback. This is a live experimental project, not a finalized academic version.
# ai-biz2026-classifier
