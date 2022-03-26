from utils import smooth_one_hot

import torch
import torch.nn.functional as F


class SCELoss(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes=10, label_smooth=0.0):
        super(SCELoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.label_smooth = label_smooth
        self.cross_entropy = torch.nn.CrossEntropyLoss(label_smoothing=label_smooth)

    def forward(self, pred, labels):
        # CCE
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        if self.label_smooth == 0.0:
            label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        else:
            label_one_hot = smooth_one_hot(labels, self.num_classes, smoothing=self.label_smooth)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss