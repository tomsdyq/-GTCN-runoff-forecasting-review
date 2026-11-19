"""Run the published G-TCN workflow on CAMELS-US Daymet data."""
import argparse
import csv
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data.dataset import CamelsDataset
from gtcn import VARIANTS, build
from workflow import predict, test_scores, train_one_epoch, validation

PERIODS = {"train": ("1980-10-01", "1995-09-30"), "val": ("1995-10-01", "2000-09-30"),
           "test": ("2000-10-01", "2014-09-30")}


def dataset(args, ids, stage, train=None):
    return CamelsDataset(args.camels, ids, [pd.Timestamp(d) for d in PERIODS[stage]], train)


def fit(args, model, train, val, device):
    global_run = args.stage == "pretrain"
    train_loader = DataLoader(train, batch_size=512, shuffle=True, drop_last=global_run)
    val_loader = DataLoader(val, batch_size=512, drop_last=global_run)
    if not len(train_loader) or not len(val_loader):
        raise ValueError("Too few training or validation windows for this basin selection.")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    warmup = args.schedule_epochs * 0.25
    def multiplier(epoch):
        return (epoch + 1) / warmup if epoch < warmup else 0.99 ** (epoch - warmup)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    best = -np.inf
    with (args.out / "training.csv").open("w", newline="") as log:
        writer = csv.writer(log)
        writer.writerow(["epoch", "training_loss", "validation_mse", "validation_selection_score"])
        for epoch in range(1, args.epochs + 1):
            error = train_one_epoch(model, train_loader, optimizer, global_run, device)
            scheduler.step()
            mse, score = validation(model, val_loader, device)
            score = float(score)
            writer.writerow([epoch, error, mse, score])
            log.flush()
            if np.isfinite(score) and score > best:
                torch.save(model.state_dict(), args.out / "best.pt")
                best = score
            print(f"epoch {epoch}: training loss {error:.5g}, validation score {score:.5g}", flush=True)
    if not np.isfinite(best):
        raise RuntimeError("No finite validation score. Inspect the selected data and training log.")


def evaluate(args, model, test, device):
    obs, sim = predict(model, DataLoader(test, batch_size=512), device)
    obs, sim = [test.runoff_mm(x.numpy()) for x in (obs, sim)]
    rows = []
    start = 0
    for basin, n in zip(test.basins_list, test.length_ls):
        y, p = obs[start:start + n], sim[start:start + n]
        start += n
        scores = test_scores(y, p)
        for lead in range(7):
            rows.append({"basin_id": basin, "lead_days": lead + 1,
                         **{k: float(v[lead]) for k, v in scores.items()}})
    pd.DataFrame(rows).to_csv(args.out / "per_basin_test.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["pretrain", "finetune", "test"])
    parser.add_argument("--camels", required=True, type=Path)
    basins = parser.add_mutually_exclusive_group(required=True)
    basins.add_argument("--basins", type=Path, help="Text file with one gauge ID per line.")
    basins.add_argument("--basin", help="Single eight-digit gauge ID.")
    parser.add_argument("--checkpoint", type=Path, help="PyTorch model state dictionary.")
    parser.add_argument("--out", required=True, type=Path, help="New output directory.")
    parser.add_argument("--variant", default="full", choices=VARIANTS)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--schedule-epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.stage in ("finetune", "test") and args.checkpoint is None:
        parser.error("Finetuning and testing require --checkpoint.")
    if args.stage == "pretrain" and args.checkpoint is not None:
        parser.error("Pretraining starts from a fresh model.")
    if args.stage == "finetune" and not args.basin:
        parser.error("Use --basin to fine-tune one basin from the pretrained checkpoint.")
    if args.epochs < 1 or args.schedule_epochs < 1:
        parser.error("Epoch counts must be positive.")
    ids = [args.basin] if args.basin else args.basins.read_text().split()
    if not ids or len(set(ids)) != len(ids) or any(len(b) != 8 or not b.isdigit() for b in ids):
        parser.error("Expected distinct eight-digit gauge IDs.")
    if not all((args.camels / p).is_dir() for p in ("basin_mean_forcing/daymet", "usgs_streamflow")):
        parser.error("CAMELS root must contain basin_mean_forcing/daymet and usgs_streamflow.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model = build(args.variant).to(device)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    args.out.mkdir(parents=True, exist_ok=False)
    train = dataset(args, ids, "train")
    if args.stage == "test":
        evaluate(args, model, dataset(args, ids, "test", train), device)
    else:
        fit(args, model, train, dataset(args, ids, "val", train), device)


if __name__ == "__main__":
    main()

