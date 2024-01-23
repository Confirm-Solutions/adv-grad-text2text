from torch import nn


class ProjectionHead(nn.Module):
    def __init__(self, emb_size, vocab_size, ln_eps):
        super(ProjectionHead, self).__init__()
        self.ln_f = nn.LayerNorm(emb_size, ln_eps)
        self.emded = nn.Linear(emb_size, vocab_size)
        
    def forward(self, x, pr_labels=None):
        pr_logits = self.emded(self.ln_f(x))
        if pr_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(pr_logits.view(-1, pr_logits.size(-1)), pr_labels.view(-1))
            return loss
        else:
            return pr_logits.view(-1, pr_logits.size(-1))