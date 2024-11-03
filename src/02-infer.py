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

# import kaggle_evaluation.jane_street_inference_server


def load_data(cfg: TrainConfig) -> pd.DataFrame:
    # pyarrowでフォルダ内のparquetファイルをすべてLoad
    data_dir = os.path.join(cfg.dir.data_dir, "test.parquet")
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


@hydra.main(config_path="conf", config_name="infer", version_base="1.1")
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
    features_mean = np.load("features_mean.npy")
    features_mean = pd.Series(features_mean, index=feature_cols)
    df[feature_cols] = df[feature_cols].fillna(features_mean)
    del features_mean
    gc.collect()

    df = reduce_mem_usage(df=df)

    df[target_col] = 0  # dummy

    LOGGER.info(df)

    # Create loaders
    test_dataset = MarketDataset(
        df=df,
        feature_cols=feature_cols,
        target_col=target_col,
        weight_col=weight_col,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.test_batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
    )

    # Def model
    model = MLP(features=feature_cols)
    model.load_state_dict(torch.load(cfg.best_model_bin))
    model.to(device=device)

    # Infer
    preds = run_inference(dataloader=test_loader, model=model)
    sub_df = pd.DataFrame(
        {
            "row_id": df["row_id"].to_numpy(),
            "responder_6": preds,
        }
    )


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
