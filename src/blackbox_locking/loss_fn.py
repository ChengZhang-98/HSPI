import torch


def one_vs_rest_cross_entropy_loss(
    target_logits: torch.Tensor, remainder_logits_lists: list[torch.Tensor], labels: torch.Tensor, target_scale: float
):
    """
    target_logits: torch.Tensor, (batch_size, num_classes)
    remainder_logits_lists: list[torch.Tensor], [(batch_size, num_classes), ...]
    labels: torch.Tensor, (batch_size,)
    target_scale: float, the scale factor for the target loss. If None, scale = len(remainder_logits).
    """

    # minimize the cross-entropy loss of remainder_logits and maximize the cross-entropy loss of target_logits

    target_loss = torch.nn.functional.cross_entropy(target_logits, labels)
    remainder_loss = 0
    for r_logits in remainder_logits_lists:
        remainder_loss += torch.nn.functional.cross_entropy(r_logits, labels)

    loss = remainder_loss - target_scale * target_loss
    return loss


def one_vs_one_logits_difference_gain(logits1: torch.Tensor, logits2: torch.Tensor):
    assert logits1.shape == logits2.shape
    assert logits1.ndim == 2
    _, pred1 = torch.max(logits1, 1)
    _, pred2 = torch.max(logits2, 1)

    loss = torch.nn.functional.cross_entropy(logits1, pred2) + torch.nn.functional.cross_entropy(logits2, pred1)

    return loss
