import torch
from torch import nn
from models.modules import GaussianNoise


class AutoEncoder(nn.Module):
    def __init__(self, features):
        super().__init__()

        # self.noise = GaussianNoise(std=0.1)
        self.layers = nn.Sequential(
            nn.Linear(len(features), 512),
            # 各特徴量を同程度にscale -> 入れると学習が不安定になる
            # 誤差逆伝播で[-5, 5]の値がくるから正規化するとおかしくなる？
            # nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        """
        回帰nnでは出力層で活性化関数を使わずにMSEで学習することが多い
        ref: https://cvml-expertguide.net/terms/dl/layers/activation-function/
        """
        x = self.layers(x)

        return 5 * x.squeeze()


class SupervisedAutoEncoder(nn.Module):
    """
    ref: https://www.kaggle.com/code/gogo827jz/jane-street-supervised-autoencoder-mlp
    """

    def __init__(
        self,
        num_features: int,
        num_lag_features: int,
        tags: list[float] | None = None,  # (num_features, num_tags)
        tag_embed_dim: int = 4,
        tag_weight: float = 0.5,
        hidden_units: list[int] = [96, 96, 896, 448, 448, 256],
        dropout_rates: list[int] = [
            0.03527936123679956,
            0.038424974585075086,
            0.42409238408801436,
            0.10431484318345882,
            0.49230389137187497,
            0.32024444956111164,
            0.2716856145683449,
            0.4379233941604448,
        ],
    ):
        super().__init__()

        self.num_features = num_features
        self.tags = tags
        self.tag_weight = tag_weight

        num_all_features = num_features + num_lag_features

        if tags is not None:
            self.tag_embedding = nn.Linear(tags.shape[1], tag_embed_dim)
        else:
            self.tag_embedding = None

        num_input_features = num_all_features + (num_features * tag_embed_dim)

        # Encoder
        self.noise = GaussianNoise(std=0.035)
        self.encoder_dense = nn.Linear(num_input_features, hidden_units[0])
        self.encoder_activation = nn.SiLU()  # Swish

        # Decoder
        self.decoder_dropout = nn.Dropout(dropout_rates[1])
        self.decoder_dense = nn.Linear(hidden_units[0], num_all_features)

        # x_ae
        self.x_ae_dense = nn.Linear(num_all_features, hidden_units[1])
        self.x_ae_activation = nn.SiLU()
        self.x_ae_dropout = nn.Dropout(dropout_rates[2])

        # out_ae
        self.out_ae_dense = nn.Linear(hidden_units[1], 1)

        # x0 + Encoder
        concat_dim = num_all_features + hidden_units[0]
        self.concat_dropout = nn.Dropout(dropout_rates[3])

        # additional hidden layers
        self.hidden_layers = []
        prev_dim = concat_dim
        for i in range(2, len(hidden_units)):
            self.hidden_layers.extend(
                [
                    nn.Linear(prev_dim, hidden_units[i]),
                    nn.SiLU(),
                    nn.Dropout(dropout_rates[i + 2]),
                ]
            )
            prev_dim = hidden_units[i]
        self.hidden_layers = nn.Sequential(*self.hidden_layers)

        # out
        self.out_dense = nn.Linear(prev_dim, 1)

    def forward(self, x_feature, x_lag):
        x = torch.cat([x_feature, x_lag], dim=1)

        if self.tags is not None:
            tag_embed = self.tag_embedding(self.tags)

            batch_size = x.size(0)
            tag_embed_expanded = tag_embed.unsqueeze(0).expand(batch_size, -1, -1)
            tag_embed_flat = tag_embed_expanded.reshape(batch_size, -1)

            tag_embed_flat = self.tag_weight * tag_embed_flat

            x_input = torch.cat([x, tag_embed_flat], dim=1)
        else:
            x_input = x

        encoder = self.noise(x_input)
        encoder = self.encoder_dense(encoder)
        encoder = self.encoder_activation(encoder)

        decoder = self.decoder_dropout(encoder)
        decoder = self.decoder_dense(decoder)

        x_ae = self.x_ae_dense(decoder)
        x_ae = self.x_ae_activation(x_ae)
        x_ae = self.x_ae_dropout(x_ae)

        out_ae = self.out_ae_dense(x_ae)

        x_concat = torch.cat([x, encoder], dim=1)
        x_concat = self.concat_dropout(x_concat)

        x_hidden = self.hidden_layers(x_concat)

        out = self.out_dense(x_hidden)

        return decoder, 5 * out_ae.squeeze(), 5 * out.squeeze()
