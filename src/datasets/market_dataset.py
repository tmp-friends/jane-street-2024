import torch


class MarketDataset:
    def __init__(
        self,
        df,
        feature_cols: list[str],
        lag_cols: list[str],
        target_col: str,
        weight_col: str,
    ):
        self.features = df[feature_cols].values
        self.lags = df[lag_cols].values
        self.target = df[target_col].values
        self.weight = df[weight_col].values

    def __len__(self):
        return len(self.target)

    def __getitem__(self, ix):
        return {
            "features": torch.tensor(self.features[ix], dtype=torch.float),
            "lags": torch.tensor(self.lags[ix], dtype=torch.float),
            "target": torch.tensor(self.target[ix], dtype=torch.float),
            "weight": torch.tensor(self.weight[ix], dtype=torch.float),
        }
