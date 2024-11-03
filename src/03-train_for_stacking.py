import logging
import os
from pathlib import Path
import gc
from tqdm import tqdm
import time
import copy
from collections import defaultdict

import hydra

import numpy as np
import pyarrow.parquet as pq
import pandas as pd
import polars as pl
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from conf.type import TrainConfig
from utils.utils import set_seed
from utils.score import score_weighted_r2
from datasets.market_dataset import MarketDataset
from models.mlp import MLP


def load_data(cfg: TrainConfig) -> pd.DataFrame:
    # pyarrowでフォルダ内のparquetファイルをすべてLoad
    data_dir = os.path.join(cfg.dir.data_dir, "train.parquet")
    dataset = pq.ParquetDataset(data_dir)
    table = dataset.read()

    df = table.to_pandas()

    return df


def reduce_mem_usage(df: pd.DataFrame):
    """
    メモリ削減のためにデータ型を列ごとに変換する関数
    数値列ごとに逐次変換を行うことで、メモリ不足に対応します。
    """
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024**2
    LOGGER.info(f"メモリ使用量 (変換前): {start_mem:.2f} MB")

    for col in df.columns:
        col_type = df[col].dtype

        # 数値型の場合
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()

            # 整数型の変換
            if pd.api.types.is_integer_dtype(df[col]):
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)

            # 浮動小数点数型の変換
            elif pd.api.types.is_float_dtype(df[col]):
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

        # オブジェクト型（カテゴリ型への変換が可能）
        else:
            df[col] = df[col].astype("category")

    end_mem = df.memory_usage().sum() / 1024**2
    LOGGER.info(
        f"メモリ使用量 (変換後): {end_mem:.2f} MB, 削減率: {100 * (start_mem - end_mem) / start_mem:.1f}%"
    )

    return df


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


@torch.inference_mode()
def run_inference(
    dataloader: DataLoader,
    model: nn.Module,
) -> np.ndarray:
    model.eval()
    preds = []
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        x = data["features"].to(device, dtype=torch.float)

        outputs = model(x).squeeze()
        preds.append(outputs.detach().cpu().numpy())

    return np.concatenate(preds).flatten()


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
    df = load_data(cfg)
    LOGGER.info(df)

    feature_cols = [v for v in df.columns if "feature" in v]
    target_col = "responder_6"
    weight_col = "weight"

    # Preprocessing
    # 欠損値補完 for nn
    features_mean = df[feature_cols].mean()
    df[feature_cols] = df[feature_cols].fillna(features_mean)
    np.save("features_mean.npy", features_mean.values)
    del features_mean
    gc.collect()

    # TODO: targetを標準化
    # scaler = StandardScaler()
    # # df[feature_cols] = scaler.fit_transform(df[feature_cols].values)
    # df[target_col] = scaler.fit_transform(
    #     df[target_col].values.reshape(-1, 1)
    # ).flatten()

    df = reduce_mem_usage(df=df)

    LOGGER.info(df)

    cfg.T_max = (
        df.shape[0]
        * (cfg.n_folds - 1)
        * cfg.n_epochs
        // cfg.train_batch_size
        // cfg.n_folds
    )

    oof_preds = np.zeros(len(df))
    model_filenames = [""] * len(df)
    kf = KFold(n_splits=cfg.n_folds)
    # gkf = GroupKFold(n_splits=cfg.n_folds)
    for i, (train_ix, valid_ix) in enumerate(
        kf.split(
            X=df[feature_cols],
            y=df[target_col],
            # groups=df["date_id"],
        )
    ):
        LOGGER.info(f"Fold {i}")
        df.loc[valid_ix, "fold"] = int(i)

        # Create loaders
        train_dataset = MarketDataset(
            df=df.iloc[train_ix],
            feature_cols=feature_cols,
            target_col=target_col,
            weight_col=weight_col,
        )
        valid_dataset = MarketDataset(
            df=df.iloc[valid_ix],
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
        model = MLP(features=feature_cols)
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
        for epoch in range(cfg.n_epochs):
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
        model_filename = "fold{:.0f}_R2{:.4f}_Loss{:.4f}_epoch{:.0f}.bin".format(
            i, best_epoch_r2, best_epoch_loss, best_epoch
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

        # oof preds
        model.load_state_dict(best_model_wts)
        oof_preds[valid_ix] = run_inference(model, valid_loader)
        for ix in valid_ix:
            model_filenames[ix] = model_filename

        # Monitor
        save_history(history=history)

    oof_preds_df = pd.DataFrame(
        {
            "date_id": df["date_id"],
            "time_id": df["time_id"],
            "symbol_id": df["symbol_id"],
            "responder_6": df["responder_6"],
            "fold": df["fold"],
            "oof_preds": oof_preds,
            "model_filename": model_filenames,
        }
    )
    oof_preds_df.to_csv("oof_preds.csv", index=False)


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
