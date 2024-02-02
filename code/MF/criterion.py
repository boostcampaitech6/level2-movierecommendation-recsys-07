import torch


def get_criterion(pred: torch.Tensor, target: torch.Tensor, args):
    if args.loss_function.lower() == "bce":
        loss = torch.nn.BCEWithLogitsLoss(reduction="none")
        loss = loss(pred, target)
        loss = torch.mean(loss)
    elif args.loss_function.lower() == "roc_star":
        loss = roc_star_paper(pred, target, args)

    return loss


def roc_star_paper(y_pred, _y_true, args):
    y_true = _y_true >= 0.5

    # if batch is either all true or false return small random stub value.
    if torch.sum(y_true) == 0 or torch.sum(y_true) == y_true.shape[0]:
        return torch.tensor(y_pred.shape[0] * 1e-8, requires_grad=True)

    pos = y_pred[y_true]
    neg = y_pred[~y_true]

    ln_pos = pos.shape[0]
    ln_neg = neg.shape[0]

    max_pos = 1000  # Max number of positive training samples
    max_neg = 1000  # Max number of positive training samples
    pos = pos[torch.rand_like(pos) < max_pos / ln_pos]
    neg = neg[torch.rand_like(neg) < max_neg / ln_neg]

    ln_pos = pos.shape[0]
    ln_neg = neg.shape[0]

    pos_expand = pos.view(-1, 1).expand(-1, ln_neg).reshape(-1)
    neg_expand = neg.repeat(ln_pos)

    diff = -(pos_expand - neg_expand - args.gamma)
    diff = diff[diff > 0]

    loss = torch.sum(diff * diff)
    loss = loss / (ln_pos + ln_neg)
    return loss + 1e-8