## Overview

### Description

[Jane Street](https://www.janestreet.com/)

金融市場のモデリングにおける主な課題

- fat-tailed 分布
- non-stationary time series (非定常時系列)
- 市場行動の突然の変化

### Evaluation

### Data

- responder_6を最大6ヶ月先まで予測

- train.parquet
    - date_id, time_id: 時系列の値。time_id間の実際の時間間隔は異なる場合がある。
    - symbol_id: 一意の金融商品(financial instrument)
    - weight: scoring funcの重み
    - feature_{00...78}: 匿名化されたマーケットデータ
    - responder_{0...8}: 匿名化されたリターンデータ [-5, 5]。特にresponder_6を予測したい。
- test.parquet
    - Python evaluation APIで評価
        - testデータを(date_id, time_id)毎に提供
    - is_scored
- lag.parquet
    - responder_{0...8}の値はdate_idが一つ遅れる
    - evaluation APIはdate_idの最初のtime_idで、そのdate_idに対して遅れているresponder全体を提供する
