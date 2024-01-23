from torch import nn


class ProjectionHead(nn.Module):
    def __init__(self):
        super(ProjectionHead, self).__init__()
        self.emded = nn.Linear(768, 50257)
        
    def forward(self, x, pr_labels=None):
        pr_logits = self.emded(x)
        if pr_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(pr_logits.view(-1, pr_logits.size(-1)), pr_labels.view(-1))
            return loss
        else:
            return pr_logits.view(-1, pr_logits.size(-1))