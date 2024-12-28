import logging
import os
import sys
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
import torch
import torch.nn as nn
import torch.optim as optim
from schedulefree import RAdamScheduleFree
from torch.utils.data import DataLoader

from conf.type import TrainConfig
from utils.utils import set_seed
from utils.score import score_weighted_r2
from datasets.market_dataset import MarketDataset
from models.autoencoder import SupervisedAutoEncoder


def criterion_decoder(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # loss * weight にしたいので、batch単位でloss集計をしないreduction="none"にする
    return nn.MSELoss(reduction="none")(outputs, targets)


def criterion_ae(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # loss * weight にしたいので、batch単位でloss集計をしないreduction="none"にする
    return nn.MSELoss(reduction="none")(outputs, targets)


def criterion(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # loss * weight にしたいので、batch単位でloss集計をしないreduction="none"にする
    return nn.MSELoss(reduction="none")(outputs, targets)


def train_one_epoch(
    cfg: TrainConfig,
    dataloader: DataLoader,
    model: nn.Module,
    optimizer: optim,
    epoch: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.train()
    optimizer.train()

    dataset_size = 0
    running_loss = 0.0

    y_true = []
    y_preds = []
    weights = []
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        x_feature = data["features"].to(device, dtype=torch.float)
        x_lag = data["lags"].to(device, dtype=torch.float)
        y = data["target"].to(device, dtype=torch.float)
        weight = data["weight"].to(device, dtype=torch.float)

        x = torch.cat([x_feature, x_lag], dim=1)

        batch_size = x.size(0)

        decoder, out_ae, out = model(x_feature, x_lag)

        # (batch_size, num_features) のため、num_featuresで平均をとる
        loss_decoder = criterion_decoder(decoder, x).mean(dim=1)
        loss_out_ae = criterion_ae(out_ae, y)  # (batch_size,)
        loss = criterion(out, y)  # (batch_size,)

        loss_decoder = (weight * loss_decoder).mean()
        loss_out_ae = (weight * loss_out_ae).mean()
        loss = (weight * loss).mean()

        loss /= cfg.num_accumulates

        # 同一の計算グラフから複数回 backward() を呼ぶと勾配が累積される
        loss_decoder.backward(retain_graph=True)
        loss_out_ae.backward(retain_graph=True)
        loss.backward()

        if (step + 1) % cfg.num_accumulates == 0:
            optimizer.step()

            # zero the parameter gradients
            optimizer.zero_grad()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

        y_true.append(y.detach())
        y_preds.append(out.detach())
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
    optimizer.eval()

    dataset_size = 0
    running_loss = 0.0

    y_true = []
    y_preds = []
    weights = []
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        x_feature = data["features"].to(device, dtype=torch.float)
        x_lag = data["lags"].to(device, dtype=torch.float)
        y = data["target"].to(device, dtype=torch.float)
        weight = data["weight"].to(device, dtype=torch.float)

        x = torch.cat([x_feature, x_lag], dim=1)

        batch_size = x.size(0)

        decoder, out_ae, out = model(x_feature, x_lag)

        # (batch_size, num_features) のため、num_featuresで平均をとる
        loss_decoder = criterion_decoder(decoder, x).mean(dim=1)
        loss_out_ae = criterion_ae(out_ae, y)  # (batch_size,)
        loss = criterion(out, y)  # (batch_size,)

        loss_decoder = (weight * loss_decoder).mean()
        loss_out_ae = (weight * loss_out_ae).mean()
        loss = (weight * loss).mean()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

        y_true.append(y.detach())
        y_preds.append(out.detach())
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


@hydra.main(config_path="conf", config_name="train", version_base="1.1")
def main(cfg: TrainConfig):
    """
    ref: Neural Network Starter Pytorch Version (https://www.kaggle.com/code/a763337092/neural-network-starter-pytorch-version)
    """
    #####################
    # Load data
    #####################

    df: pl.LazyFrame = pl.scan_parquet(os.path.join(cfg.dir.data_dir, "train.parquet"))

    tags_df = pl.read_csv(os.path.join(cfg.dir.data_dir, "features.csv"))
    tags_array = tags_df.drop("feature").to_pandas().values.astype(np.float32)
    tags_tensor = torch.from_numpy(tags_array).to(device)

    #####################
    # Feature engineering
    #####################

    # 前日のresponder_idの値との差分をlag特徴量とする
    # ref: https://www.kaggle.com/code/motono0223/js24-preprocessing-create-lags
    lag_cols_original = ["date_id", "symbol_id"] + [f"responder_{i}" for i in range(9)]
    lag_cols_rename = {f"responder_{i}": f"responder_{i}_lag_1" for i in range(9)}

    lags_df = df.select(pl.col(lag_cols_original)).rename(lag_cols_rename)
    lags_df = lags_df.with_columns(date_id=pl.col("date_id") + 1)  # lagged by 1day
    # pick up last record of prev date
    lags_df = lags_df.group_by(["date_id", "symbol_id"], maintain_order=True).last()

    df = df.join(lags_df, on=["date_id", "symbol_id"], how="left")

    #####################
    # Preprocessing
    #####################

    feature_cols = [v for v in df.collect_schema() if "feature" in v]
    lag_cols = [v for v in df.collect_schema() if "lag" in v]
    all_feature_cols = feature_cols + lag_cols
    target_col = "responder_6"
    weight_col = "weight"

    # 時系列でsplit (valid にする date_id は固定)
    train_date_point = 820
    valid_date_point = 1634
    train_df = df.filter(
        train_date_point <= pl.col("date_id"), pl.col("date_id") <= valid_date_point
    )
    valid_df = df.filter(pl.col("date_id") > valid_date_point)

    del df
    gc.collect()

    # 欠損値補完
    train_df = train_df.fill_null(strategy="forward").fill_null(0)
    valid_df = valid_df.fill_null(strategy="forward").fill_null(0)

    train_df: pd.DataFrame = train_df.collect().to_pandas()
    valid_df: pd.DataFrame = valid_df.collect().to_pandas()

    LOGGER.info(train_df)

    # Create loaders
    train_dataset = MarketDataset(
        df=train_df,
        feature_cols=feature_cols,
        lag_cols=lag_cols,
        target_col=target_col,
        weight_col=weight_col,
    )
    valid_dataset = MarketDataset(
        df=valid_df,
        feature_cols=feature_cols,
        lag_cols=lag_cols,
        target_col=target_col,
        weight_col=weight_col,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.valid_batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    # Def model
    model = SupervisedAutoEncoder(
        num_features=len(feature_cols),
        num_lag_features=len(lag_cols),
    )
    model.to(device=device)

    optimizer = RAdamScheduleFree(model.parameters(), lr=cfg.lr, betas=(0.9, 0.999))

    # Train
    start = time.time()
    best_epoch = -np.inf
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch_loss = -np.inf
    best_epoch_r2 = -np.inf
    early_stopping_cnt = 0

    history = defaultdict(list)
    for epoch in range(1, cfg.num_epochs + 1):
        train_epoch_loss, train_epoch_r2 = train_one_epoch(
            cfg=cfg,
            dataloader=train_loader,
            model=model,
            optimizer=optimizer,
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

        if best_epoch_r2 <= valid_epoch_r2:
            LOGGER.info(f"Val R2 Improved ({best_epoch_r2} ---> {valid_epoch_r2})")

            best_epoch = epoch
            best_epoch_loss = valid_epoch_loss
            best_epoch_r2 = valid_epoch_r2
            best_model_wts = copy.deepcopy(model.state_dict())

            early_stopping_cnt = 0

        else:
            early_stopping_cnt += 1
            if early_stopping_cnt >= cfg.early_stopping_steps:
                LOGGER.info(f"Early stopping at Epoch {epoch}")
                break

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
