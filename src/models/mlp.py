from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, features):
        super().__init__()

        self.batch_norm0 = nn.BatchNorm1d(len(features))
        self.dropout = nn.Dropout(0.1)

        self.dense1 = nn.Linear(len(features), 384)
        self.batch_norm1 = nn.BatchNorm1d(384)
        self.dropout1 = nn.Dropout(0.1)

        self.dense2 = nn.Linear(384, 896)
        self.batch_norm2 = nn.BatchNorm1d(896)
        self.dropout2 = nn.Dropout(0.2)

        self.dense3 = nn.Linear(896, 896)
        self.batch_norm3 = nn.BatchNorm1d(896)
        self.dropout3 = nn.Dropout(0.2)

        self.dense4 = nn.Linear(896, 394)
        self.batch_norm4 = nn.BatchNorm1d(394)
        self.dropout4 = nn.Dropout(0.2)

        self.dense5 = nn.Linear(394, 1)

        self.relu = nn.ReLU(inplace=True)
        # self.prelu = nn.PReLU()
        # self.leaky_relu = nn.LeakyReLU(nagative_slope=0.01, inplace=True)
        # self.gelu = nn.GELU()
        # self.rrelu = nn.RReLU()

    def forward(self, x):
        x = self.batch_norm0(x)
        x = self.dropout(x)

        x = self.dense1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        x = self.dense2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.dropout2(x)

        x = self.dense3(x)
        x = self.batch_norm3(x)
        x = self.relu(x)
        x = self.dropout3(x)

        x = self.dense4(x)
        x = self.batch_norm4(x)
        x = self.relu(x)
        x = self.dropout4(x)

        x = self.dense5(x)

        return x
