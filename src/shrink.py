import os

import cv2
import pandas as pd

from src.preprocess import load_hand

SRC_DIR = r"data\archive\boneage-training-dataset\boneage-training-dataset"
OUT_DIR = r"data\small"

os.makedirs(OUT_DIR, exist_ok=True)

ids = pd.read_csv(r"data\archive\boneage-training-dataset.csv")["id"].tolist()

for n, i in enumerate(ids, 1):
    out_path = os.path.join(OUT_DIR, f"{int(i)}.png")
    if not os.path.exists(out_path):          # skip ones already done
        small = load_hand(os.path.join(SRC_DIR, f"{int(i)}.png"), 512)
        cv2.imwrite(out_path, small)
    if n % 1000 == 0:
        print(f"{n} of {len(ids)} done")

print("Finished!")