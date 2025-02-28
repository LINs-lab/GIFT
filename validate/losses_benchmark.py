import os
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



def SoftCrossEntropy(inputs, target, reduction='average'):
    input_log_likelihood = -F.log_softmax(inputs, dim=1)
    target_log_likelihood = F.softmax(target, dim=1)
    batch = inputs.shape[0]
    loss = torch.sum(torch.mul(input_log_likelihood, target_log_likelihood)) / batch
    return loss

def batch_label_smoothing(labels, num_classes, alpha=0.1):
    batch_size = labels.size(0)
    smoothed_labels = torch.full((batch_size, num_classes), alpha / (num_classes - 1)).cuda()
    smoothed_labels.scatter_(1, labels.unsqueeze(1), 1.0 - alpha)
    return smoothed_labels

def cosine_loss(output, teacher_target):
    cosine_sim = F.cosine_similarity(output, teacher_target, dim=1) 
    sum_cosine_sim = cosine_sim.sum()
    loss_all = 1 - sum_cosine_sim / output.shape[0]
    return loss_all


loss_functions = {
    'mse': nn.MSELoss().cuda(),
    'ce': nn.CrossEntropyLoss().cuda(),
    'kl': nn.KLDivLoss(reduction="batchmean"),
    'softce': SoftCrossEntropy, 
}


def compute_loss(criterion_name, output, teacher_target, hard_target, loss_hyper=0.1):
    T = 20
    
    if criterion_name  == 'cosine':
        loss_all = cosine_loss(output, teacher_target)

    if criterion_name  == 'gift':
        num_classes = output.shape[1]
        teacher_target = F.normalize(teacher_target, dim=1)
        smoothed_labels = batch_label_smoothing(hard_target, num_classes, alpha=0.1)
        smoothed_labels = F.normalize(smoothed_labels, dim=1)
        modified_labels = teacher_target + 0.1 * smoothed_labels
        loss_all = cosine_loss(output, modified_labels)

        
    if criterion_name == 'mse' or criterion_name == 'ce' or criterion_name == 'softce':
        loss_all = loss_functions[criterion_name](output, teacher_target if criterion_name != 'ce' else hard_target)
        
    if criterion_name == 'mse_ce':
        loss_mse = loss_functions['mse'](output, teacher_target)
        loss_ce = loss_hyper * loss_functions['ce'](output, hard_target)
        loss_all = loss_mse + loss_ce
    
    if criterion_name == 'softce_ce':
        loss_softce = loss_functions['softce'](output, teacher_target)
        loss_ce = loss_hyper * loss_functions['ce'](output, hard_target)
        loss_all = loss_softce + loss_ce

    if criterion_name == 'kl':
        student_logits = F.log_softmax(output / T, dim=1)
        teacher_logits = F.softmax(teacher_target / T, dim=1)
        loss_all = loss_functions['kl'](student_logits, teacher_logits)
    
    return loss_all