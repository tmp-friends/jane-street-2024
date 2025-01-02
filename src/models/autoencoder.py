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
        # num_categories: int = 40,
        # category_emb_dim: int = 16,
        # max_date_id: int = 968,
        # date_emb_dim: int = 16,
        hidden_units=[96, 96, 896, 448, 448, 256],
        dropout_rates=[
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

        num_all_features = num_features + num_lag_features
        # num_all_features = (
        #     num_features + num_lag_features + category_emb_dim + date_emb_dim
        # )

        # if self.category_emb_dim > 0:
        #     self.category_embedding = nn.Embedding(
        #         num_embeddings=num_categories,
        #         embedding_dim=category_emb_dim,
        #     )
        # else:
        #     self.category_embedding = None

        # if self.date_emb_dim > 0:
        #     self.date_embedding = nn.Embedding(
        #         num_embeddings=max_date_id,
        #         embedding_dim=date_emb_dim,
        #     )
        # else:
        #     self.date_embedding = None

        self.input_norm = nn.BatchNorm1d(num_all_features)

        # Encoder
        self.noise = GaussianNoise(std=0.1)
        self.encoder_dense = nn.Linear(num_all_features, hidden_units[0])
        self.encoder_norm = nn.BatchNorm1d(hidden_units[0])
        self.encoder_activation = nn.SiLU()  # Swish

        # Decoder
        self.decoder_dropout = nn.Dropout(dropout_rates[1])
        self.decoder_dense = nn.Linear(hidden_units[0], num_all_features)

        # x_ae
        self.x_ae_dense = nn.Linear(num_all_features, hidden_units[1])
        self.x_ae_norm = nn.BatchNorm1d(hidden_units[1])
        self.x_ae_activation = nn.SiLU()
        self.x_ae_dropout = nn.Dropout(dropout_rates[2])

        # out_ae - multi label
        self.out_ae_dense = nn.Linear(hidden_units[1], 9)

        # x0 + Encoder
        concat_dim = num_all_features + hidden_units[0]
        self.concat_norm = nn.BatchNorm1d(concat_dim)
        self.concat_dropout = nn.Dropout(dropout_rates[3])

        # additional hidden layers
        self.hidden_layers = []
        prev_dim = concat_dim
        for i in range(2, len(hidden_units)):
            self.hidden_layers.extend(
                [
                    nn.Linear(prev_dim, hidden_units[i]),
                    nn.BatchNorm1d(hidden_units[i]),
                    nn.SiLU(),
                    nn.Dropout(dropout_rates[i + 2]),
                ]
            )
            prev_dim = hidden_units[i]
        self.hidden_layers = nn.Sequential(*self.hidden_layers)

        # out - multi label
        self.out_dense_6 = nn.Linear(prev_dim, 1)
        self.out_dense_others = nn.Linear(prev_dim, 8)

    def forward(self, x_feature, x_lag, x_category=None, x_date=None):
        x = torch.cat([x_feature, x_lag], dim=1)

        # if (self.category_embedding is not None) and (x_category is not None):
        #     category_embed = self.categorical_embedding(x_category)
        #     x = torch.cat([x, category_embed], dim=1)

        # if (self.date_embedding is not None) and (x_date is not None):
        #     date_embed = self.date_embedding(x_date)
        #     x = torch.cat([x, date_embed], dim=1)

        x0 = self.input_norm(x)

        if self.training:
            encoder_input = self.noise(x0)
        else:
            encoder_input = x0

        encoder = self.encoder_dense(encoder_input)
        encoder = self.encoder_norm(encoder)
        encoder = self.encoder_activation(encoder)

        decoder = self.decoder_dropout(encoder)
        decoder = self.decoder_dense(decoder)

        x_ae = self.x_ae_dense(decoder)
        x_ae = self.x_ae_norm(x_ae)
        x_ae = self.x_ae_activation(x_ae)
        x_ae = self.x_ae_dropout(x_ae)

        out_ae = self.out_ae_dense(x_ae)

        x_concat = torch.cat([x0, encoder], dim=1)
        x_concat = self.concat_norm(x_concat)
        x_concat = self.concat_dropout(x_concat)

        x_hidden = self.hidden_layers(x_concat)

        out_6 = self.out_dense_6(x_hidden)
        out_others = self.out_dense_others(x_hidden)

        # responder_6 を index=6 に挟む
        out_before6 = out_others[:, :6]  # (batch_size, 6)
        out_after6 = out_others[:, 6:]  # (batch_size, 2)
        out = torch.cat([out_before6, out_6, out_after6], dim=1)  # (batch_size, 10)

        return decoder, 5 * out_ae.squeeze(), 5 * out.squeeze()
