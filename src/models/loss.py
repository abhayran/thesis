import torch


def dice_loss_torch(pred, gnd, smooth: float = 1.):
    pred = pred.contiguous().view(pred.shape[0], -1)
    gnd = gnd.contiguous().view(gnd.shape[0], -1)
    num = torch.sum(torch.mul(pred, gnd), dim=1) + smooth
    den = torch.sum(pred.pow(2) + gnd.pow(2), dim=1) + smooth
    return (1 - num / den).mean()
