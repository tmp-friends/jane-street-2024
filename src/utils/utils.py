import os
import random

import numpy as np
import pandas as pd
import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
    """
    メモリ削減のためにデータ型を列ごとに変換
    数値列ごとに逐次変換を行うことで、メモリ不足に対応
    """
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024**2
    print(f"メモリ使用量 (変換前): {start_mem:.2f} MB")

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
    print(
        f"メモリ使用量 (変換後): {end_mem:.2f} MB, 削減率: {100 * (start_mem - end_mem) / start_mem:.1f}%"
    )

    return df
