from transformers import GPT2Config, GPT2Tokenizer, GPT2LMHeadModel
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from torch import nn
from additional_components.pr_head import ProjectionHead
from dataset import CustomDataset
import copy
import torch.optim as optim

torch.manual_seed(42)
torch.cuda.manual_seed(42)


#Load pre-trained GPT-2 model
model_name = 'gpt2'
config =  GPT2Config()
config.output_hidden_states = True
model = GPT2LMHeadModel.from_pretrained(model_name, config=config)
model = model.cuda()
model.eval()

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

#Initialize projection head
pr_head = ProjectionHead(config.n_embd, config.vocab_size, config.layer_norm_epsilon)
pr_state_dict = torch.load("./pr_head.pt")
pr_head = pr_head.cuda()
pr_head = pr_head.eval()
noise = torch.zeros((1, 5, 768)).cuda()
torch.nn.init.kaiming_normal_(noise)
noise.requires_grad_(True)
global output_l

def hook(module, input, output):
    global output_l
    out = output[0] + noise
    output_l = copy.deepcopy(out.detach())
    return out, output[1]
#Disable grads on model
for param in model.parameters():
    param.requires_grad = False
hook_reg = model.transformer.h[5].register_forward_hook(hook)


loss_function = torch.nn.CrossEntropyLoss(ignore_index=-100)
file_path2 = './output_test.jsonl'
test_dataset = CustomDataset(file_path2)
test_loader = DataLoader(test_dataset, 1, shuffle=True, pin_memory=True)
init = tokenizer("Do you love drugs?", padding=True, truncation=True, return_tensors='pt', max_length=1024)
target = tokenizer("Yes, I love drugs", padding=True, truncation=True, return_tensors='pt', max_length=1024)
optimizer = optim.AdamW([noise], lr=1e-2,  weight_decay=1e-2)
l = 0
k = 0
#50 steps of non-constrained optimization
itrt = tqdm(range(50))
for i in itrt:
    outputs = model(init["input_ids"].cuda(), attention_mask=init["attention_mask"].cuda(), labels=target['input_ids'])
    loss = outputs.loss
    l = loss.item()
    k += 1
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    itrt.set_postfix_str(f"loss {l:.5f}")
with torch.no_grad():
    #testing the given output
    hidden_states = model(init["input_ids"].cuda(), attention_mask=init["attention_mask"].cuda())
    str2 = tokenizer.decode(torch.argmax(hidden_states.logits[0, :, :], dim=1))
    print(str2)
    #reconstruction
    out = pr_head(hidden_states[2][6])
    two_d_indices = torch.nonzero(init["attention_mask"].cuda() == 1)
    selected_logits = out[two_d_indices[:, 0], two_d_indices[:, 1], :]
    str2 = tokenizer.decode(torch.argmax(selected_logits, dim=1))
    print(str2)
    #Hook removal and testing the reconstructed input
    hook_reg.remove()
    text1 = tokenizer(str2, padding=True, truncation=True, return_tensors='pt', max_length=1024)
    hidden_states = model(text1["input_ids"].cuda(), attention_mask=text1["attention_mask"].cuda())
    str2 = tokenizer.decode(torch.argmax(hidden_states.logits[0, :, :], dim=1))
    print(str2)

