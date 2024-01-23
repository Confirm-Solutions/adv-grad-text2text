from transformers import GPT2Config, GPT2Tokenizer, GPT2LMHeadModel
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from torch import nn
from additional_components.pr_head import ProjectionHead
from dataset import CustomDataset

# Load pre-trained GPT-2 model and tokenizer
model_name = 'gpt2'
config =  GPT2Config()
model = GPT2LMHeadModel.from_pretrained(model_name, config, output_hidden_states=True)
model = model.cuda()
model.eval()

# Input text
seed = 42
np.random.seed(seed)
torch.random.manual_seed(seed)
torch.cuda.manual_seed(seed)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pr_head = ProjectionHead(config.n_embd, config.vocab_size, config.layer_norm_epsilon)
pr_head = pr_head.cuda()
for param in model.parameters():
    param.requires_grad = False
optim = torch.optim.Adam(pr_head.parameters(), 1e-3)
    
file_path = './output_train.jsonl'
custom_dataset = CustomDataset(file_path)
loader = DataLoader(custom_dataset, 8, shuffle=True, pin_memory=True)

file_path2 = './output_test.jsonl'
test_dataset = CustomDataset(file_path2)
test_loader = DataLoader(test_dataset, 5, shuffle=False, pin_memory=True)

f = open("./results.csv", "w")
f.write("attention_block,token_acc\n")
#Training
for j in range(1, 11):
    print(f"Attention Block {j}")
    for i in range(2):
        l_mean = 0
        k = 0
        itrt = tqdm(loader)
        itrt.set_description(f"Epoch {i}")
        for text in itrt:
            text = tokenizer(text, padding=True, truncation=True, return_tensors='pt', max_length=1024)
            with torch.no_grad():
                hidden_states = model(text["input_ids"].cuda(), attention_mask=text["attention_mask"].cuda())
            loss = pr_head(hidden_states[2][j].detach().clone(), pr_labels=text["input_ids"].cuda())
            del hidden_states
            optim.zero_grad()
            loss.backward()
            optim.step()
            l_mean += loss.cpu().item()
            k+=1
            itrt.set_postfix_str(f"{l_mean/k:.3f}")
    itrt2 = tqdm(test_loader)
    acc_sum = 0
    l = 0
    for text2 in itrt2:
        text2 = tokenizer(text2, padding=True, truncation=True, return_tensors='pt', max_length=1024)
        with torch.no_grad():
            hidden_states = model(text2["input_ids"].cuda(), attention_mask=text2["attention_mask"].cuda())
            out = pr_head(hidden_states[2][1].detach().clone())
            del hidden_states
            acc_sum += (torch.argmax(out, dim=1) ==  text2["input_ids"].cuda().view(-1)).sum().cpu().item()
            l += text2["input_ids"].view(-1).shape[0]
            itrt2.set_postfix_str(f"{acc_sum/l:.4f}")
    f.write(f"{j},{acc_sum/l:.4f}\n")
#Testing
