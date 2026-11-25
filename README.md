# Bone Age Prediction from Hand X-rays

Predicts a child's bone age (in months) from a hand X-ray and the child's sex.

## What I built

- A pretrained ResNet-50 looks at the X-ray and produces 2048 features.
- The child's sex goes through a small layer of 32 features.
- Both are joined and passed through a small head (512 -> 1) that outputs the age in months.
- Training uses augmentation (small rotations, shifts, brightness changes), SmoothL1 loss and the AdamW optimizer.
- Images are padded to a square and resized to 512x512.
- Training ran on a free Google Colab T4 GPU: 10 epochs, batch size 16, about 9 minutes per epoch.

## Results

The metric is MAE (mean absolute error) in months. Lower is better.

| Model | MAE (months) |
|---|---|
| Baseline: always guess the average age | 34.5 |
| My model (overall) | 6.94 |
| My model (boys) | 6.74 |
| My model (girls) | 7.18 |
| Contest winner | 4.2 |

My model is about 5x better than the baseline, but still behind the contest winner.

Data split: 10,088 training images and 2,523 validation images (random 80/20 split).
Full numbers are in `results/results.json`.

## Project structure

- `notebooks/01_explore.ipynb` - data exploration, train/validation split, baseline
- `src/preprocess.py` - loads an X-ray, pads it to a square, resizes it
- `src/dataset.py` - dataset class with augmentation
- `src/model.py` - the model (ResNet-50 + sex input)
- `src/train.py` - training script
- `src/shrink.py` - makes smaller 512x512 copies of the images
- `splits/` - the train/validation split and statistics
- `results/results.json` - final scores

## How to run

1. Get the hand X-ray dataset yourself (see the note below).
2. Install Python 3 and the main libraries (PyTorch, torchvision, pandas, Pillow).
3. Run the training script:

    python -m src.train --img_dir PATH_TO_IMAGES --out_dir PATH_TO_OUTPUT --epochs 10 --batch_size 16 --size 512 --num_workers 2

A GPU is strongly recommended. I trained on Google Colab.

## Note about data

The X-ray images are medical data, so they are **not included** in this repository. Only code and results are shared.