import torch


def score_weighted_r2(y_true, y_pred, weights):
    numerator = torch.sum(weights * (y_true - y_pred) ** 2)
    denominator = torch.sum(weights * (y_true**2))
    r2 = 1 - (numerator / denominator)

    return r2.item()
