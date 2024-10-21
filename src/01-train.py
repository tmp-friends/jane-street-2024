import logging
import os
from pathlib import Path
import gc
from tqdm import tqdm

import hydra

import numpy as np
import pyarrow.parquet as pq
import pandas as pd
from matplotlib import pyplot as plt

from utils.utils import set_seed
from conf.type import TrainConfig


@hydra.main(config_path="conf", config_name="train", version_base="1.1")
def main(cfg: TrainConfig):
    # Load data
    data_dir = os.path.join(cfg.dir.data_dir, "train.parquet")
    dataset = pq.ParquetDataset(data_dir)
    table = dataset.read()
    df = table.to_pandas()
    print(df)

    # Create fold

    # Create dataloader

    # Def model

    # Train

    # Monitor


if __name__ == "__main__":
    # Logger
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s:%(name)s - %(message)s"
    )
    LOGGER = logging.getLogger(Path(__file__).name)

    # For descriptive error messages
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    set_seed()

    main()
