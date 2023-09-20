import torch
from torch import nn
import torch.nn.functional as F

from transformers import AutoModel


class RadBERTMultiClassMultiLabel(nn.Module):
    """
    RadBERTMultiClassMultiLabel: Model expects batches of natural language sentences, will
    classify reports with multiple label
    """
    def __init__(self, num_classes, checkpoint):
        super().__init__()
        self.num_classes = num_classes
        self.checkpoint = checkpoint

        self.transformer_encoder = AutoModel.from_pretrained(self.checkpoint)
        self.transformer_encoder_hidden_size = self.transformer_encoder.config.hidden_size
        self.linear_classifier = nn.Linear(self.transformer_encoder_hidden_size, self.num_classes)
    
    def forward(self, x):
        encoder_out = self.transformer_encoder(**x)
        logits = self.linear_classifier(encoder_out.last_hidden_state[:, 0, :])
        return logits

class MultiClassMultiLabelLoss(nn.Module):
    def __init__(self, uncertain_label, device=None):
        super(MultiClassMultiLabelLoss, self).__init__()
        self.uncertain_label = uncertain_label
        self.device = device
    
    def forward(self, output, target, penalize_certainity=False):
        certain_mask = (target != self.uncertain_label)
        loss_func = nn.MultiLabelSoftMarginLoss(weight=certain_mask.type(torch.float))
        if not penalize_certainity:
            return loss_func(output, target)
        else:
            certainity_loss_func = nn.MultiLabelSoftMarginLoss(weight=(~certain_mask).type(torch.float))
            ones = torch.ones(output.shape)
            zeros = torch.zeros(output.shape)
            if self.device is not None:
                ones = torch.ones(output.shape, device=self.device)
                zeros = torch.zeros(output.shape, device=self.device)
            return loss_func(output, target) + 0.1 * (certainity_loss_func(output, ones) + certainity_loss_func(output, zeros))