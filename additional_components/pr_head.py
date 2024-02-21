from torch import nn
import torch


class ProjectionHead(nn.Module):
    def __init__(self, emb_size, vocab_size, ln_eps):
        super(ProjectionHead, self).__init__()
        self.ln_f = nn.LayerNorm(emb_size, ln_eps)
        self.emded = nn.Sequential(nn.Linear(emb_size, emb_size),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(emb_size, vocab_size))
        
    def forward(self, x, pr_labels=None, mask=0):
        pr_logits = self.emded(self.ln_f(x))
        if pr_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            two_d_indices = torch.nonzero(mask == 1)
            selected_logits = pr_logits[two_d_indices[:, 0], two_d_indices[:, 1], :]
            selected_labels = pr_labels[two_d_indices[:, 0], two_d_indices[:, 1]]
            loss = loss_fct(selected_logits, selected_labels)
            return loss
        else:
            return pr_logits