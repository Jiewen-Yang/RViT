import numpy as np

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def topN_params_init():
    top1 = AverageMeter()
    top5 = AverageMeter()
    return top1, top5


def Multi_Accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def update_topN(outputs, labels, top1, top3):
    prec1, prec5 = Multi_Accuracy(outputs.data, labels, topk=(1, 5))
    top1.update(prec1.item(), labels.size(0))
    top5.update(prec5.item(), labels.size(0))
    return top1, top5

