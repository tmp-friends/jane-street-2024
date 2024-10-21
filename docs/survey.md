## Survey

### Discussion

#### Useful references and starter materials
https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting/discussion/540437

- 過去コンペ

### Jane Street Market Prediction

#### Jane Street: EDA of day 0 and feature importance
https://www.kaggle.com/code/carlmcbrideellis/jane-street-eda-of-day-0-and-feature-importance

-

#### Neural Network Starter Pytorch Version
https://www.kaggle.com/code/a763337092/neural-network-starter-pytorch-version

- [nn starter notebook](https://www.kaggle.com/code/gogo827jz/jane-street-neural-network-starter)のPyTorch版

#### Yirun's Solution (1st place): Training Supervised Autoencoder with MLP
https://www.kaggle.com/competitions/jane-street-market-prediction/discussion/224348

- 1st place solution
- training notebook: https://www.kaggle.com/code/gogo827jz/jane-street-supervised-autoencoder-mlp
- MLP + XGBoost
- MLP part
    - 教師ありAutoEncoder
    - [公開notebook](https://www.kaggle.com/code/aimind/bottleneck-encoder-mlp-keras-tuner-8601c5)ではCV分割の前に1つの教師ありAutoEncoderを別々に学習させていたが、labelのleakを引き起こしていた
    - 教師ありAutoEncoderとMLPを各CVにおいて1つのモデルで学習させるようにした
