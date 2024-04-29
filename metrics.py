import torch


def dcg(y_true: torch.Tensor, y_pred: torch.Tensor, k: int, rel: str) -> torch.Tensor:
    sorted_idx = torch.argsort(y_pred, descending=True, dim=-1)[:, :k]
    y_true_sorted = torch.gather(y_true, -1, sorted_idx)
    if rel == "exp2":
        gain = torch.pow(2, y_true_sorted) - 1.0
    else:
        gain = y_true_sorted
    return gain / torch.log2(torch.arange(1, k + 1, 1) + 1)


def ndcgk(
    targets: torch.Tensor, predictions: torch.Tensor, k: int, rel: str
) -> torch.Tensor:
    pdcg = dcg(targets, predictions, k, rel).sum(-1)
    idcg = dcg(targets, targets, k, rel).sum(-1) + 1e-10
    ndcg = pdcg / idcg
    return ndcg
