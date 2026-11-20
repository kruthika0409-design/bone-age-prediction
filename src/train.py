import argparse
import json
import os
import time

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import BoneAgeDataset
from src.model import BoneAgeNet


def evaluate(model, loader, device, std):
    """The exam: returns the error in months (overall, boys, girls)."""
    model.eval()
    preds, targets, sexes = [], [], []
    with torch.no_grad():
        for x, male, y in loader:
            out = model(x.to(device), male.to(device))
            preds.append(out.cpu())
            targets.append(y)
            sexes.append(male)
    preds, targets, sexes = torch.cat(preds), torch.cat(targets), torch.cat(sexes)

    err = (preds - targets).abs() * std          # turn back into months
    result = {"mae": err.mean().item()}
    if (sexes == 1).any():
        result["mae_boys"] = err[sexes == 1].mean().item()
    if (sexes == 0).any():
        result["mae_girls"] = err[sexes == 0].mean().item()
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--img_dir", required=True)
    p.add_argument("--out_dir", default="results")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--subset", type=int, default=0, help="0 = use all data")
    p.add_argument("--num_workers", type=int, default=0)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Training on:", device)

    stats = json.load(open("splits/stats.json"))
    std = stats["STD"]
    train_df = pd.read_csv("splits/train.csv")
    val_df = pd.read_csv("splits/val.csv")

    if args.subset:
        train_df = train_df.sample(args.subset, random_state=0)
        val_df = val_df.sample(min(len(val_df), max(args.subset // 4, 8)), random_state=0)

    train_ds = BoneAgeDataset(train_df, args.img_dir, train=True, size=args.size)
    val_ds = BoneAgeDataset(val_df, args.img_dir, train=False, size=args.size)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)

    model = BoneAgeNet().to(device)
    loss_fn = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_mae = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        start = time.time()
        total = 0.0
        for x, male, y in train_dl:
            x, male, y = x.to(device), male.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x, male), y)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(y)

        train_loss = total / len(train_ds)
        res = evaluate(model, val_dl, device, std)
        print(f"Epoch {epoch}/{args.epochs}  loss {train_loss:.4f}  "
              f"exam MAE {res['mae']:.1f} months  ({time.time() - start:.0f}s)")

        if res["mae"] < best_mae:
            best_mae = res["mae"]
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best.pt"))
            res["epoch"] = epoch
            res["baseline_mae"] = stats["baseline_mae"]
            with open(os.path.join(args.out_dir, "results.json"), "w") as f:
                json.dump(res, f, indent=2)

    print("Best exam MAE:", round(best_mae, 1), "months")


if __name__ == "__main__":
    main()