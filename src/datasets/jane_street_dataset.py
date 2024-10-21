import torch


class JaneStreetDataset:
    def __init__(self, df, features):
        self.features = df[features].values
        self.label = (df["resp"] > 0).astype("int").values.reshape(-1, 1)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, ix):
        return {
            "features": torch.tensor(self.features[ix], dtype=torch.float),
            "label": torch.tensor(self.label[ix], dtype=torch.float),
        }
