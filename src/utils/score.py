import torch


def score_weighted_r2(
    y_true: torch.Tensor, y_pred: torch.Tensor, weights: torch.Tensor
) -> float:
    """
    Calculate the sample weighted zero-mean R-squared score using torch tensors.

    Args:
        y_true (torch.Tensor): Ground-truth values for responder_6.
        y_pred (torch.Tensor): Predicted values for responder_6.
        weights (torch.Tensor): Sample weight tensor.

    Returns:
        float: The weighted zero-mean R-squared score.
    """
    numerator = torch.sum(weights * (y_true - y_pred) ** 2)
    denominator = torch.sum(weights * y_true**2)

    r2_score = 1 - numerator / denominator

    return r2_score.item()
