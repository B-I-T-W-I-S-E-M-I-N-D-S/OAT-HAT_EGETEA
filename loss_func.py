# import torch
# import numpy as np
# import torch.nn as nn
# import torch.nn.functional as F

# class MultiCrossEntropyLoss(nn.Module):
#     def __init__(self, focal=False, weight=None, reduce=True):
#         super(MultiCrossEntropyLoss, self).__init__()
#         self.focal = focal
#         self.weight= weight
#         self.reduce = reduce

#     def forward(self, input, target):
#         #IN: input: unregularized logits [B, C] target: multi-hot representaiton [B, C]
#         target_sum = torch.sum(target, dim=1)
#         target_div = torch.where(target_sum != 0, target_sum, torch.ones_like(target_sum)).unsqueeze(1)
#         target = target/target_div
#         logsoftmax = nn.LogSoftmax(dim=1).to(input.device)
#         if not self.focal:
#             if self.weight is None:
#                 output = torch.sum(-target * logsoftmax(input), 1)
#             else:
#                 output = torch.sum(-target * logsoftmax(input) /self.weight, 1)
#         else:
#             softmax = nn.Softmax(dim=1).to(input.device)
#             p = softmax(input)
#             output = torch.sum(-target * (1 - p)**2 * logsoftmax(input), 1)
            
#         if self.reduce:
#             return torch.mean(output)
#         else:
#             return output
    

# def cls_loss_func(y,output, use_focal=False, weight=None, reduce=True):
#     input_size=y.size()
#     y = y.float().cuda()
#     if weight is not None:
#         weight = weight.cuda()
#     loss_func = MultiCrossEntropyLoss(focal=use_focal, weight=weight, reduce=reduce)
    
#     y=y.reshape(-1,y.size(-1))
#     output=output.reshape(-1,output.size(-1))
#     loss = loss_func(output,y)
    
#     if not reduce:
#         loss = loss.reshape(input_size[:-1])
    
#     return loss


# def regress_loss_func(y,output):
#     y = y.float().cuda()
    
#     #y=y.unsqueeze(-1)
#     y=y.reshape(-1,y.size(-1))
#     output=output.reshape(-1,output.size(-1))
    
#     bgmask= y[:,1] < -1e2
    
#     fg_logits = output[~bgmask]
#     bg_logits = output[bgmask]
    
#     fg_target = y[~bgmask]
#     bg_target = y[bgmask]
    
#     loss = nn.functional.l1_loss(fg_logits,fg_target)
        
#     if(loss.isnan()):
#         return torch.tensor([0.0], requires_grad=True).cuda()
#     return loss


# def suppress_loss_func(y,output):
#     y = y.float().cuda()
    
#     #y=y.unsqueeze(-1)
#     y=y.reshape(-1,y.size(-1))
#     output=output.reshape(-1,output.size(-1))

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from functools import partial
        
class MultiCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes=None, focal=False, weight=None, reduce=True):
        super(MultiCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes  # Will be set dynamically if None
        self.focal = focal
        self.weight = weight
        self.reduce = reduce
        self.gamma_f = 0.05
        
        # Initialize with default size, will be resized if needed
        default_classes = num_classes if num_classes is not None else 23
        self.gamma_ = torch.zeros(default_classes).cuda() + 0.025
        self.register_buffer('pos_grad', torch.zeros(default_classes-1).cuda())
        self.register_buffer('neg_grad', torch.zeros(default_classes-1).cuda())
        self.register_buffer('pos_neg', torch.ones(default_classes-1).cuda())

    def _initialize_buffers(self, num_classes):
        """Resize buffers when we know the number of classes"""
        if self.num_classes is None:
            self.num_classes = num_classes
            
        if self.gamma_.size(0) != num_classes:
            # Resize existing buffers instead of registering new ones
            self.gamma_ = torch.zeros(num_classes).cuda() + 0.025
            self.pos_grad = torch.zeros(num_classes-1).cuda()
            self.neg_grad = torch.zeros(num_classes-1).cuda()
            self.pos_neg = torch.ones(num_classes-1).cuda()

    def forward(self, input, target):
        # Dynamically determine number of classes from input
        num_classes = input.size(-1)
        self._initialize_buffers(num_classes)
        
        target_sum = torch.sum(target, dim=1)
        target_div = torch.where(target_sum != 0, target_sum, torch.ones_like(target_sum)).unsqueeze(1)
        target = target/target_div
        logsoftmax = nn.LogSoftmax(dim=1).to(input.device)
        gamma = self.gamma_.clone()
        
        # Make sure gamma has the right size and is on the right device
        if gamma.size(0) != num_classes:
            gamma = torch.zeros(num_classes).to(input.device) + 0.025
        
        # Only modify gamma if we have pos_neg buffer and it has the right size
        if self.pos_neg is not None and len(self.pos_neg) == num_classes - 1:
            gamma[:-1] = gamma[:-1] + self.gamma_f * (1 - self.pos_neg)

        if not self.focal:
            if self.weight is None:
                output = torch.sum(-target * logsoftmax(input), 1)
            else:
                output = torch.sum(-target * logsoftmax(input) /self.weight, 1)
        else:
            softmax = nn.Softmax(dim=1).to(input.device)
            p = softmax(input)
            
            # Expand gamma to match the batch dimension for broadcasting
            gamma_expanded = gamma.unsqueeze(0).expand(p.size(0), -1)
            output = torch.sum(-target * (1 - p)**gamma_expanded * logsoftmax(input), 1)

        if self.reduce:
            return torch.mean(output)
        else:
            return output
        
    def map_func(self, x, s):
        min_val = torch.min(x)
        max_val = torch.max(x)
        mu = torch.mean(x)
        x = (x - min_val) / (max_val - min_val)
        return 1 / (1 + torch.exp(-s * (x - mu)))
    
    def collect_grad(self, target, grad):
        grad = torch.abs(grad.reshape(-1, grad.shape[-1])).cuda()
        target = target.reshape(-1, target.shape[-1]).cuda()
        
        # Ensure buffers have the right size
        if self.pos_grad.size(0) != target.size(-1) - 1:
            return  # Skip if buffer size doesn't match
            
        pos_grad = torch.sum(grad * target, dim=0)[:-1]
        neg_grad = torch.sum(grad * (1 - target), dim=0)[:-1]
        self.pos_grad += pos_grad
        self.neg_grad += neg_grad
        self.pos_neg = torch.clamp(self.pos_grad / (self.neg_grad + 1e-10), min=0, max=1)
        self.pos_neg = self.map_func(self.pos_neg, 1)
    

def cls_loss_func(y, output, use_focal=False, weight=None, reduce=True):
    input_size = y.size()
    y = y.float().cuda()
    if weight is not None:
        weight = weight.cuda()
    
    # Get number of classes from the target tensor
    num_classes = y.size(-1)
    loss_func = MultiCrossEntropyLoss(num_classes=num_classes, focal=use_focal, weight=weight, reduce=reduce)
    
    y = y.reshape(-1, y.size(-1))
    output = output.reshape(-1, output.size(-1))
    loss = loss_func(output, y)
    
    if not reduce:
        loss = loss.reshape(input_size[:-1])
    
    return loss


def cls_loss_func_(loss_func, y, output, use_focal=False, weight=None, reduce=True):
    input_size = y.size()
    y = y.float().cuda()
    if weight is not None:
        weight = weight.cuda()
    
    y = y.reshape(-1, y.size(-1))
    output = output.reshape(-1, output.size(-1))
    loss = loss_func(output, y)
    
    if not reduce:
        loss = loss.reshape(input_size[:-1])
    
    return loss

def regress_loss_func(y, output):
    y = y.float().cuda()
    y = y.reshape(-1, y.size(-1))
    output = output.reshape(-1, output.size(-1))
    
    bgmask = y[:, 1] < -1e2
    
    fg_logits = output[~bgmask]
    bg_logits = output[bgmask]
    
    fg_target = y[~bgmask]
    bg_target = y[bgmask]
    
    if fg_logits.numel() == 0:  # No foreground samples
        return torch.tensor([0.0], requires_grad=True).cuda()
    
    loss = nn.functional.l1_loss(fg_logits, fg_target)
        
    if loss.isnan():
        return torch.tensor([0.0], requires_grad=True).cuda()
    return loss


def suppress_loss_func(y, output):
    y = y.float().cuda()
    y = y.reshape(-1, y.size(-1))
    output = output.reshape(-1, output.size(-1))
    loss = nn.functional.binary_cross_entropy(output,y)
        
    return loss
