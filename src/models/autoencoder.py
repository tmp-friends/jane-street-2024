import math
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


class PositionalEncoding(torch.nn.Module):
    def __init__(self, embedding_dim, max_len=968):
        super().__init__()
        self.embedding_dim = embedding_dim

        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * (-math.log(max_len) / embedding_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x.long()
        return self.pe[x]


class SupervisedAutoEncoder(nn.Module):
    """
    ref: https://www.kaggle.com/code/gogo827jz/jane-street-supervised-autoencoder-mlp
    """

    def __init__(
        self,
        num_features: int,
        num_lag_features: int,
        num_categories: int = 40,
        category_emb_dim: int = 8,
        max_date_id: int = 968,
        date_emb_dim: int = 16,
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

        # 数値+lag の次元
        self.num_base_features = num_features + num_lag_features

        # カテゴリ & 日付 埋め込み
        self.category_emb_dim = category_emb_dim
        self.date_emb_dim = date_emb_dim

        self.category_embedding = nn.Embedding(
            num_embeddings=num_categories,
            embedding_dim=category_emb_dim,
        )
        self.date_embedding = PositionalEncoding(
            embedding_dim=date_emb_dim,
            max_len=max_date_id,
        )

        # Embedding 後の入力次元
        # base_features + category_emb_dim + date_emb_dim
        self.all_input_dim = self.num_base_features + category_emb_dim + date_emb_dim

        # # 入力正規化
        self.input_norm = nn.BatchNorm1d(self.all_input_dim)

        # ----------------------------------------------------
        # Encoder
        # ----------------------------------------------------
        self.noise = GaussianNoise(std=0.1)
        self.encoder_dense = nn.Linear(self.all_input_dim, hidden_units[0])
        self.encoder_norm = nn.BatchNorm1d(hidden_units[0])
        self.encoder_activation = nn.SiLU()  # Swish

        # ----------------------------------------------------
        # Decoder
        #   今回は「数値＋lag のみ」再構築すると想定 -> 出力を num_base_features に設定
        # ----------------------------------------------------
        self.decoder_dropout = nn.Dropout(dropout_rates[1])
        self.decoder_dense = nn.Linear(hidden_units[0], self.num_base_features)

        # ----------------------------------------------------
        # x_ae
        #   Decoder の出力をさらに隠れ層へ
        # ----------------------------------------------------
        self.x_ae_dense = nn.Linear(self.num_base_features, hidden_units[1])
        self.x_ae_norm = nn.BatchNorm1d(hidden_units[1])
        self.x_ae_activation = nn.SiLU()
        self.x_ae_dropout = nn.Dropout(dropout_rates[2])

        # out_ae (1次元)
        self.out_ae_dense = nn.Linear(hidden_units[1], 1)

        # ----------------------------------------------------
        # x0 + Encoder concat
        # ----------------------------------------------------
        concat_dim = self.all_input_dim + hidden_units[0]
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

        # 最終 out (1次元)
        self.out_dense = nn.Linear(prev_dim, 1)

    def forward(self, x_feature, x_lag, x_category, x_date):
        """
        x_feature:  (batch_size, num_features)       [float]
        x_lag:      (batch_size, num_lag_features)   [float]
        x_category: (batch_size,) or (batch_size, 1) [int → embedding index]
        x_date:     (batch_size,) or (batch_size, 1) [int → embedding index]
        """
        # 1) 数値部
        x_num = torch.cat([x_feature, x_lag], dim=1)  # (batch_size, num_base_features)

        # 2) カテゴリ埋め込み
        #    x_category: (batch_size,) (int)
        cat_embed = self.category_embedding(x_category)  # (batch_size, cat_emb_dim)

        # 3) 日付埋め込み
        date_embed = self.date_embedding(x_date)  # (batch_size, date_emb_dim)

        # 4) 結合
        x_input = torch.cat([x_num, cat_embed, date_embed], dim=1)

        x0 = self.input_norm(x_input)

        # Gaussian Noise (train時のみ)
        if self.training:
            encoder_input = self.noise(x0)
        else:
            encoder_input = x0

        # Encoder
        encoder = self.encoder_dense(encoder_input)
        encoder = self.encoder_norm(encoder)
        encoder = self.encoder_activation(encoder)

        # Decoder
        decoder = self.decoder_dropout(encoder)
        decoder = self.decoder_dense(decoder)

        # x_ae
        x_ae = self.x_ae_dense(decoder)
        x_ae = self.x_ae_norm(x_ae)
        x_ae = self.x_ae_activation(x_ae)
        x_ae = self.x_ae_dropout(x_ae)

        out_ae = self.out_ae_dense(x_ae)

        # concat
        x_concat = torch.cat([x0, encoder], dim=1)
        x_concat = self.concat_norm(x_concat)
        x_concat = self.concat_dropout(x_concat)

        x_hidden = self.hidden_layers(x_concat)

        out = self.out_dense(x_hidden)

        return decoder, 5.0 * out_ae.squeeze(), 5.0 * out.squeeze()
