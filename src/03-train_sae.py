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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from conf.type import TrainConfig
from utils.utils import set_seed, reduce_mem_usage
from utils.score import score_weighted_r2
from datasets.market_dataset import MarketDataset
from models.autoencoder import SupervisedAutoEncoder

# 全データの平均と分散
features_mean = {
    "feature_00": 0.640198826789856,
    "feature_01": 0.03755598142743111,
    "feature_02": 0.6368075609207153,
    "feature_03": 0.6365063786506653,
    "feature_04": 0.013741530478000641,
    "feature_05": -0.02173694409430027,
    "feature_06": -0.006415014620870352,
    "feature_07": -0.010971736162900925,
    "feature_08": -0.04653771221637726,
    "feature_09": 32.596106194690265,
    "feature_10": 4.95929203539823,
    "feature_11": 167.6541592920354,
    "feature_12": -0.13415881991386414,
    "feature_13": -0.07573335617780685,
    "feature_14": -0.12015637010335922,
    "feature_15": -0.7470195889472961,
    "feature_16": -0.6257441639900208,
    "feature_17": -0.7294047474861145,
    "feature_18": -0.042215555906295776,
    "feature_19": -0.08798160403966904,
    "feature_20": -0.15741558372974396,
    "feature_21": 0.10528526455163956,
    "feature_22": 0.018054703250527382,
    "feature_23": 0.03165541961789131,
    "feature_24": 2.733017921447754,
    "feature_25": 0.39958420395851135,
    "feature_26": -0.11045943945646286,
    "feature_27": -0.5332594513893127,
    "feature_28": -0.4522790312767029,
    "feature_29": -0.5739678144454956,
    "feature_30": -0.7905704975128174,
    "feature_31": 0.10600688308477402,
    "feature_32": 0.40044134855270386,
    "feature_33": -0.021725023165345192,
    "feature_34": 0.4226262867450714,
    "feature_35": 0.42143046855926514,
    "feature_36": -0.00023802756913937628,
    "feature_37": 0.027961043640971184,
    "feature_38": 0.010258913040161133,
    "feature_39": 0.005768273025751114,
    "feature_40": 0.017485467717051506,
    "feature_41": 0.038347117602825165,
    "feature_42": -0.06123563274741173,
    "feature_43": -0.11644423753023148,
    "feature_44": -0.12342483550310135,
    "feature_45": -0.028769943863153458,
    "feature_46": -0.015200662426650524,
    "feature_47": 0.015717582777142525,
    "feature_48": -0.0033910537604242563,
    "feature_49": -0.0052393232472240925,
    "feature_50": -0.2285808026790619,
    "feature_51": -0.3548349440097809,
    "feature_52": -0.358092725276947,
    "feature_53": 0.2607136368751526,
    "feature_54": 0.18796788156032562,
    "feature_55": 0.3154229521751404,
    "feature_56": -0.1471923440694809,
    "feature_57": 0.15730056166648865,
    "feature_58": -0.021774644032120705,
    "feature_59": -0.0037768862675875425,
    "feature_60": -0.010220836848020554,
    "feature_61": -0.03178725391626358,
    "feature_62": -0.3769100308418274,
    "feature_63": -0.3229374587535858,
    "feature_64": -0.3718394339084625,
    "feature_65": -0.10233989357948303,
    "feature_66": -0.13688170909881592,
    "feature_67": -0.14402112364768982,
    "feature_68": -0.06875362992286682,
    "feature_69": -0.11862917989492416,
    "feature_70": -0.11789549142122269,
    "feature_71": -0.06013699993491173,
    "feature_72": -0.10766122490167618,
    "feature_73": -0.09921672940254211,
    "feature_74": -0.10233042389154434,
    "feature_75": -0.05991339311003685,
    "feature_76": -0.06349952518939972,
    "feature_77": -0.07424316555261612,
    "feature_78": -0.07759837061166763,
}
features_std = {
    "feature_00": 1.027751088142395,
    "feature_01": 1.0967519283294678,
    "feature_02": 1.0156300067901611,
    "feature_03": 1.0170334577560425,
    "feature_04": 1.0726385116577148,
    "feature_05": 0.9639211297035217,
    "feature_06": 1.0963259935379028,
    "feature_07": 1.0789952278137207,
    "feature_08": 0.7962697148323059,
    "feature_09": 23.72976726545254,
    "feature_10": 3.1867162933797224,
    "feature_11": 163.44513161352285,
    "feature_12": 0.6700984835624695,
    "feature_13": 0.5805172920227051,
    "feature_14": 0.664044201374054,
    "feature_15": 0.37517768144607544,
    "feature_16": 0.3393096327781677,
    "feature_17": 0.3603287935256958,
    "feature_18": 0.9911752939224243,
    "feature_19": 1.0550744533538818,
    "feature_20": 0.6643751263618469,
    "feature_21": 0.38239365816116333,
    "feature_22": 0.950261116027832,
    "feature_23": 0.8119344711303711,
    "feature_24": 1.4362775087356567,
    "feature_25": 1.0947270393371582,
    "feature_26": 1.077124834060669,
    "feature_27": 1.0645726919174194,
    "feature_28": 1.0676648616790771,
    "feature_29": 0.2640742361545563,
    "feature_30": 0.19689509272575378,
    "feature_31": 0.3815343976020813,
    "feature_32": 1.2996565103530884,
    "feature_33": 0.9989405870437622,
    "feature_34": 1.3409572839736938,
    "feature_35": 1.3365675210952759,
    "feature_36": 0.8695492148399353,
    "feature_37": 0.7334080934524536,
    "feature_38": 0.698810338973999,
    "feature_39": 0.7965824604034424,
    "feature_40": 0.518515944480896,
    "feature_41": 0.6384949088096619,
    "feature_42": 0.8168442249298096,
    "feature_43": 0.5228385925292969,
    "feature_44": 0.6521403193473816,
    "feature_45": 0.8666537404060364,
    "feature_46": 0.9039222002029419,
    "feature_47": 3.2711963653564453,
    "feature_48": 0.6570901274681091,
    "feature_49": 0.7083076238632202,
    "feature_50": 1.0132617950439453,
    "feature_51": 0.6081287860870361,
    "feature_52": 0.9250587224960327,
    "feature_53": 1.0421689748764038,
    "feature_54": 0.5859629511833191,
    "feature_55": 0.9191848039627075,
    "feature_56": 0.9549097418785095,
    "feature_57": 1.0204777717590332,
    "feature_58": 0.8327276110649109,
    "feature_59": 0.8309783339500427,
    "feature_60": 0.8389413356781006,
    "feature_61": 1.192766547203064,
    "feature_62": 1.388945460319519,
    "feature_63": 0.09957146644592285,
    "feature_64": 0.3396177291870117,
    "feature_65": 1.01683509349823,
    "feature_66": 1.0824761390686035,
    "feature_67": 0.642227828502655,
    "feature_68": 0.5312599539756775,
    "feature_69": 0.6208390593528748,
    "feature_70": 0.6724499464035034,
    "feature_71": 0.5356909036636353,
    "feature_72": 0.6534596681594849,
    "feature_73": 1.0855497121810913,
    "feature_74": 1.0880277156829834,
    "feature_75": 1.2321789264678955,
    "feature_76": 1.2345560789108276,
    "feature_77": 1.0921478271484375,
    "feature_78": 1.0924347639083862,
}


def fetch_scheduler(cfg: TrainConfig, optimizer: optim) -> lr_scheduler:
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
    else:
        return None

    return scheduler


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

        decoder, out_ae, out = model(x)

        # (batch_size, num_features) のため、num_featuresで平均をとる
        loss_decoder = criterion_decoder(decoder, x).mean(dim=1)
        loss_out_ae = criterion_ae(out_ae, y)  # (batch_size,)
        loss = criterion(out, y)  # (batch_size,)

        loss_decoder = (weight * loss_decoder).mean()
        loss_out_ae = (weight * loss_out_ae).mean()
        loss = (weight * loss).mean()

        loss /= cfg.n_accumulates

        # 同一の計算グラフから複数回 backward() を呼ぶと勾配が累積される
        loss_decoder.backward(retain_graph=True)
        loss_out_ae.backward(retain_graph=True)
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

        decoder, out_ae, out = model(x)

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
    # ref: https://www.kaggle.com/code/motono0223/js24-preprocessing-create-lags
    df = df.filter(pl.col("date_id") >= 1100)

    # lags_df: pl.LazyFrame = pl.scan_parquet

    feature_cols = [v for v in df.collect_schema() if "feature" in v]
    target_col = "responder_6"
    weight_col = "weight"

    # Preprocessing
    # features_mean_df = (
    #     df.select([pl.col(v).mean().alias(v) for v in feature_cols]).collect().row(0)
    # )
    # features_std_df = (
    #     df.select([pl.col(v).std().alias(v) for v in feature_cols]).collect().row(0)
    # )
    # features_mean = {v: features_mean_df[i] for i, v in enumerate(feature_cols)}
    # features_std = {v: features_std_df[i] for i, v in enumerate(feature_cols)}

    # 欠損値補完
    # df = df.with_columns(
    #     [pl.col(v).fill_null(features_mean[v]) for i, v in enumerate(feature_cols)]
    # )
    df = df.fill_null(strategy="forward").fill_null(0)
    # Normalize
    df = df.with_columns(
        [
            ((pl.col(v) - features_mean[v]) / features_std[v]).alias(v)
            for v in feature_cols
        ]
    )

    # save
    joblib.dump({"mean": features_mean, "std": features_std}, "scaler.pkl")

    df: pd.DataFrame = df.collect().to_pandas()
    # df = reduce_mem_usage(df)
    LOGGER.info(df)

    cfg.T_max = df.shape[0] * (5 - 1) * cfg.n_epochs // cfg.train_batch_size // 5

    valid_date_point = 1634
    train_df = df[df["date_id"] < valid_date_point]
    valid_df = df[df["date_id"] >= valid_date_point]

    # Create loaders
    train_dataset = MarketDataset(
        df=train_df,
        feature_cols=feature_cols,
        target_col=target_col,
        weight_col=weight_col,
    )
    valid_dataset = MarketDataset(
        df=valid_df,
        feature_cols=feature_cols,
        target_col=target_col,
        weight_col=weight_col,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train_batch_size,
        shuffle=False,
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
    model = SupervisedAutoEncoder(num_features=len(feature_cols))
    model.to(device=device)

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = fetch_scheduler(cfg=cfg, optimizer=optimizer)

    # Train
    start = time.time()
    best_epoch = -np.inf
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch_loss = -np.inf
    best_epoch_r2 = -np.inf
    early_stopping_cnt = 0

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
