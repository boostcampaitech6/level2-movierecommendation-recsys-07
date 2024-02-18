import torch


def get_criterion(pred: torch.Tensor, target: torch.Tensor, args):
    """if args.loss_function.name.lower() == "bce":
        loss = torch.nn.BCEWithLogitsLoss(reduction="none")
        loss = loss(pred, target)
        loss = torch.mean(loss)
    elif args.loss_function.name.lower() == "roc_star":
        loss = roc_star_paper(pred, target, args)
    elif args.loss_function.name.lower() == "bpr":
        loss = BPR(pred, target, args)"""
    try:
        loss_function_name = args.loss_function.name.lower()
        loss_function = {"original_ease": original_ease}.get(loss_function_name)
        loss = loss_function(pred, target, args)
    except:
        raise NotImplementedError(
            f"loss function {args.loss_function} is not implemented"
        )
    return loss


def roc_star_paper(y_pred: torch.Tensor, _y_true: torch.Tensor, args):
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

    diff = -(pos_expand - neg_expand - args.loss_function.gamma)
    diff = diff[diff > 0]

    loss = torch.sum(diff * diff)
    loss = loss / (ln_pos + ln_neg)
    return loss + 1e-8


def BPR(pred: torch.Tensor, target: torch.Tensor, args):
    """
    Bayesian Personalized Ranking loss
    batch가 유저 단위로 구성되지 않기에 정확한 BRP은 아님."""
    y_true = target > 0.5

    # if batch is either all true or false return small random stub value.
    if torch.sum(y_true) == 0 or torch.sum(y_true) == target.shape[0]:
        return torch.tensor(target.shape[0] * 1e-8, requires_grad=True)

    pos = pred[y_true]
    neg = pred[~y_true]

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

    diff = -(pos_expand - neg_expand)
    diff = diff[diff > 0]

    loss = -torch.sum(torch.log(torch.sigmoid(diff))) / (ln_pos + ln_neg)

    return loss


def original_ease(recon_x, x, args):
    loss = x.sub(recon_x)
    loss = loss**2
    loss = loss.sum()
    regularization = torch.norm(recon_x)
    return loss + args.loss_function.lambd * regularization


def loss_function_vae(recon_x, x, mu, logvar, anneal=1.0):
    BCE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
    KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

    return BCE + anneal * KLD


def loss_function_dae(recon_x, x):
    BCE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
    return BCE
