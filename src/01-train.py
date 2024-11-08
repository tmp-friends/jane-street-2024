import logging
import os
from pathlib import Path
import gc
from tqdm import tqdm
import time
import copy
from collections import defaultdict
import joblib

import hydra

import numpy as np
import pandas as pd
import polars as pl
from matplotlib import pyplot as plt
from sklearn.model_selection import GroupKFold
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from conf.type import TrainConfig
from utils.utils import set_seed, reduce_mem_usage
from utils.score import score_weighted_r2
from datasets.market_dataset import MarketDataset
from models.mlp import MLP
from models.lstm import LSTM


def fetch_scheduler(cfg: TrainConfig, optimizer: optim) -> lr_scheduler:
    if cfg.scheduler is None:
        return None

    if cfg.scheduler == "CosineAnnealingLR":
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.T_max,
            eta_min=cfg.min_lr,
        )
    elif cfg.scheduler == "CosineAnnealingWarmRestarts":
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=cfg.T_0,
            eta_min=cfg.min_lr,
        )

    return scheduler


def criterion(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return nn.MSELoss()(outputs, targets)


def train_one_epoch(
    cfg: TrainConfig,
    dataloader: DataLoader,
    model: nn.Module,
    optimizer: optim,
    scheduler: lr_scheduler,
    epoch: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.train()

    dataset_size = 0
    running_loss = 0.0

    y_true = []
    y_preds = []
    weights = []
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        x = data["features"].to(device, dtype=torch.float)
        y = data["target"].to(device, dtype=torch.float)
        weight = data["weight"].to(device, dtype=torch.float)

        batch_size = x.size(0)

        outputs = model(x).squeeze()

        loss = criterion(outputs, y)
        loss /= cfg.n_accumulates

        loss.backward()

        if (step + 1) % cfg.n_accumulates == 0:
            optimizer.step()

            # zero the parameter gradients
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

        y_true.append(y.detach())
        y_preds.append(outputs.detach())
        weights.append(weight.detach())

        bar.set_postfix(
            Epoch=epoch,
            LR=optimizer.param_groups[0]["lr"],
            Loss=loss.item(),
        )

    epoch_loss = running_loss / dataset_size

    y_true = torch.cat(y_true)
    y_preds = torch.cat(y_preds)
    weights = torch.cat(weights)
    epoch_r2 = score_weighted_r2(y_true=y_true, y_pred=y_preds, weights=weights)

    return epoch_loss, epoch_r2


@torch.inference_mode()
def valid_one_epoch(
    dataloader: DataLoader,
    model: nn.Module,
    optimizer: optim,
    epoch: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()

    dataset_size = 0
    running_loss = 0.0

    y_true = []
    y_preds = []
    weights = []
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        x = data["features"].to(device, dtype=torch.float)
        y = data["target"].to(device, dtype=torch.float)
        weight = data["weight"].to(device, dtype=torch.float)

        batch_size = x.size(0)

        outputs = model(x).squeeze()
        loss = criterion(outputs, y)

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

        y_true.append(y.detach())
        y_preds.append(outputs.detach())
        weights.append(weight.detach())

        bar.set_postfix(
            Epoch=epoch,
            LR=optimizer.param_groups[0]["lr"],
            Loss=loss.item(),
        )

    epoch_loss = running_loss / dataset_size

    y_true = torch.cat(y_true)
    y_preds = torch.cat(y_preds)
    weights = torch.cat(weights)
    epoch_r2 = score_weighted_r2(y_true=y_true, y_pred=y_preds, weights=weights)

    return epoch_loss, epoch_r2


def save_history(history: dict) -> None:
    history = pd.DataFrame.from_dict(history)
    history.to_csv("history.csv", index=False)

    plt.plot(range(history.shape[0]), history["Train Loss"].values, label="Train Loss")
    plt.plot(range(history.shape[0]), history["Valid Loss"].values, label="Valid Loss")
    plt.xlabel("epochs")
    plt.ylabel("Loss")
    plt.grid()
    plt.legend()
    plt.savefig("plt-loss.png")
    plt.clf()

    plt.plot(range(history.shape[0]), history["Train R2"].values, label="Train R2")
    plt.plot(range(history.shape[0]), history["Valid R2"].values, label="Valid R2")
    plt.xlabel("epochs")
    plt.ylabel("R2")
    plt.grid()
    plt.legend()
    plt.savefig("plt-r2.png")
    plt.clf()

    plt.plot(range(history.shape[0]), history["lr"].values, label="lr")
    plt.xlabel("epochs")
    plt.ylabel("lr")
    plt.grid()
    plt.legend()
    plt.savefig("plt-lr.png")
    plt.clf()


@hydra.main(config_path="conf", config_name="train", version_base="1.1")
def main(cfg: TrainConfig):
    """
    ref: Neural Network Starter Pytorch Version (https://www.kaggle.com/code/a763337092/neural-network-starter-pytorch-version)
    """
    # Load data
    df: pl.LazyFrame = pl.scan_parquet(os.path.join(cfg.dir.data_dir, "train.parquet"))
    df = df.filter(pl.col("date_id") >= 1455)  # 1year に絞る

    feature_cols = [v for v in df.collect_schema() if "feature" in v]
    target_col = "responder_6"
    weight_col = "weight"

    # Preprocessing
    features_mean = (
        df.select([pl.col(v).mean().alias(v) for v in feature_cols]).collect().row(0)
    )
    features_std = (
        df.select([pl.col(v).std().alias(v) for v in feature_cols]).collect().row(0)
    )

    # 欠損値補完
    df = df.with_columns(
        [pl.col(v).fill_null(features_mean[i]) for i, v in enumerate(feature_cols)]
    )
    # Standardize
    df = df.with_columns(
        [
            ((pl.col(v) - features_mean[i]) / features_std[i]).alias(v)
            for i, v in enumerate(feature_cols)
        ]
    )

    # save
    np.save("features_mean.npy", features_mean)
    joblib.dump({"mean": features_mean, "std": features_std}, "scaler.pkl")

    df: pd.DataFrame = df.collect().to_pandas()
    df = reduce_mem_usage(df)
    LOGGER.info(df)

    cfg.T_max = (
        df.shape[0]
        * (cfg.n_folds - 1)
        * cfg.n_epochs
        // cfg.train_batch_size
        // cfg.n_folds
    )

    gkf = GroupKFold(n_splits=cfg.n_folds)
    for i, (train_ix, valid_ix) in enumerate(
        gkf.split(
            X=df[feature_cols],
            y=df[target_col],
            groups=df["date_id"],
        )
    ):
        df.loc[valid_ix, "fold"] = int(i)

    # Create loaders
    train_dataset = MarketDataset(
        df=df[df["fold"] != cfg.fold].reset_index(drop=True),
        feature_cols=feature_cols,
        target_col=target_col,
        weight_col=weight_col,
    )
    valid_dataset = MarketDataset(
        df=df[df["fold"] == cfg.fold].reset_index(drop=True),
        feature_cols=feature_cols,
        target_col=target_col,
        weight_col=weight_col,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train_batch_size,
        num_workers=4,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.valid_batch_size,
        shuffle=False,
        pin_memory=True,
    )

    # Def model
    # model = MLP(features=feature_cols)
    model = LSTM(input_size=79, hidden_dim=512, output_size=1, num_layers=1)
    model.to(device=device)

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = fetch_scheduler(cfg=cfg, optimizer=optimizer)

    # Train
    start = time.time()
    best_epoch = -np.inf
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch_loss = -np.inf
    best_epoch_r2 = -np.inf

    history = defaultdict(list)
    for epoch in range(1, cfg.n_epochs + 1):
        train_epoch_loss, train_epoch_r2 = train_one_epoch(
            cfg=cfg,
            dataloader=train_loader,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
        )
        valid_epoch_loss, valid_epoch_r2 = valid_one_epoch(
            dataloader=valid_loader,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
        )

        LOGGER.info(
            f"""
            Epoch {epoch}:
                Train R2: {train_epoch_r2:.6f} - Valid R2: {valid_epoch_r2:.6f}
                Train Loss: {train_epoch_loss:.6f} - Valid Loss: {valid_epoch_loss:.6f}
            """
        )

        history["Train Loss"].append(train_epoch_loss)
        history["Valid Loss"].append(valid_epoch_loss)
        history["Train R2"].append(train_epoch_r2)
        history["Valid R2"].append(valid_epoch_r2)
        history["lr"].append(scheduler.get_last_lr()[0])

        # deep copy the model
        if best_epoch_r2 <= valid_epoch_r2:
            LOGGER.info(f"Val R2 Improved ({best_epoch_r2} ---> {valid_epoch_r2})")

            best_epoch = epoch
            best_epoch_loss = valid_epoch_loss
            best_epoch_r2 = valid_epoch_r2
            best_model_wts = copy.deepcopy(model.state_dict())

    # Save a model file from the current directory
    model_filename = "R2{:.4f}_Loss{:.4f}_epoch{:.0f}.bin".format(
        best_epoch_r2, best_epoch_loss, best_epoch
    )
    torch.save(best_model_wts, model_filename)
    LOGGER.info("Model saved")

    end = time.time()
    time_elapsed = end - start
    LOGGER.info(
        "Training complete in {:.0f}h {:.0f}m {:.0f}s".format(
            time_elapsed // 3600,
            (time_elapsed % 3600) // 60,
            (time_elapsed % 3600) % 60,
        )
    )
    LOGGER.info("Best R2: {:.4f}".format(best_epoch_r2))

    # Monitor
    save_history(history=history)


if __name__ == "__main__":
    # Logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s:%(name)s - %(message)s",
    )
    LOGGER = logging.getLogger(Path(__file__).name)

    # For descriptive error messages
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    # Set GPU device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    set_seed()

    main()
