import torch


class MarketDataset:
    def __init__(
        self,
        df,
        feature_cols: list[str],
        lag_cols: list[str],
        category_cols: list[str],
        timeseries_cols: list[str],
        target_cols: list[str],
        weight_col: str,
    ):
        self.features = df[feature_cols].values
        self.lags = df[lag_cols].values
        self.categories = df[category_cols].values
        self.timeseries = df[timeseries_cols].values
        self.targets = df[target_cols].values
        self.weight = df[weight_col].values

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, ix):
        return {
            "features": torch.tensor(self.features[ix], dtype=torch.float),
            "lags": torch.tensor(self.lags[ix], dtype=torch.float),
            # Embedding の入力とするため int とする
            "categories": torch.tensor(self.categories[ix], dtype=torch.long),
            "timeseries": torch.tensor(self.timeseries[ix], dtype=torch.long),
            "targets": torch.tensor(self.targets[ix], dtype=torch.float),
            "weight": torch.tensor(self.weight[ix], dtype=torch.float),
        }


class SimpleMarketDataset:
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


class SimpleMarketDatasetWithDateWeight:
    def __init__(
        self,
        df,
        feature_cols: list[str],
        lag_cols: list[str],
        date_col: str,
        target_col: str,
        weight_col: str,
    ):
        self.features = df[feature_cols].values
        self.lags = df[lag_cols].values
        self.date = df[date_col].values
        self.target = df[target_col].values
        self.weight = df[weight_col].values

    def __len__(self):
        return len(self.target)

    def __getitem__(self, ix):
        return {
            "features": torch.tensor(self.features[ix], dtype=torch.float),
            "lags": torch.tensor(self.lags[ix], dtype=torch.float),
            "date": torch.tensor(self.date[ix], dtype=torch.long),
            "target": torch.tensor(self.target[ix], dtype=torch.float),
            "weight": torch.tensor(self.weight[ix], dtype=torch.float),
        }
