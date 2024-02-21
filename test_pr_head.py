from transformers import GPT2Config, GPT2Tokenizer, GPT2LMHeadModel
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from torch import nn
from additional_components.pr_head import ProjectionHead
from dataset import CustomDataset
import copy

#Load pre-trained GPT-2 model
model_name = 'gpt2'
config =  GPT2Config()
config.output_hidden_states = True
model = GPT2LMHeadModel.from_pretrained(model_name, config=config)
model = model.cuda()
model.eval()

#Set Random seed
#seed = 42
#np.random.seed(seed)
#torch.random.manual_seed(seed)
#torch.manual_seed(seed)
#torch.cuda.manual_seed(seed)

#Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

#Initialize projection head
pr_head = ProjectionHead(config.n_embd, config.vocab_size, config.layer_norm_epsilon)
pr_state_dict = torch.load("./pr_head.pt")
pr_head = pr_head.cuda()

#Disable grads on model
for param in model.parameters():
    param.requires_grad = False
def min_max_normalization(tensor):
    tensor_min = tensor.min()
    tensor_max = tensor.max()
    normalized = (tensor - tensor_min) / (tensor_max - tensor_min)
    return normalized

    
#Load datasets and loaders
file_path = './output_train.jsonl'
custom_dataset = CustomDataset(file_path)
loader = DataLoader(custom_dataset, 8, shuffle=True, pin_memory=True)

file_path2 = './output_test.jsonl'
test_dataset = CustomDataset(file_path2)
test_loader = DataLoader(test_dataset, 5, shuffle=True, pin_memory=True)
pr_head.load_state_dict(pr_state_dict)
#Create log
f = open("./results.csv", "w")
f.write("attention_block,token_acc\n")
f.flush()
itrt2 = tqdm(test_loader)
k = 0
s = 0
num_tokens = 0
for text in itrt2:
    #print(text[0][:100])
    #Training and testing per attention block
    text = tokenizer(text, padding=True, truncation=True, return_tensors='pt', max_length=1024)
    with torch.no_grad():
        hidden_states = model(text["input_ids"].cuda(), attention_mask=text["attention_mask"].cuda())
    #Reconstruct inputb and compute loss
    out = pr_head(hidden_states[2][6].detach().clone())
    two_d_indices = torch.nonzero(text["attention_mask"].cuda() == 1)
    selected_logits = out[two_d_indices[:, 0], two_d_indices[:, 1], :]
    selected_labels = text["input_ids"].cuda()[two_d_indices[:, 0], two_d_indices[:, 1]]
    str1 = tokenizer.decode(selected_labels)
    str2 = tokenizer.decode(torch.argmax(selected_logits, dim=1))
    with torch.no_grad():
        num_tokens += selected_labels.shape[0]
        text1 = tokenizer(str1, padding=True, truncation=True, return_tensors='pt', max_length=1024)
        hidden_states1 = model(text1["input_ids"].cuda(), attention_mask=text1["attention_mask"].cuda())
        text2 = tokenizer(str2, padding=True, truncation=True, return_tensors='pt', max_length=1024)
        hidden_states2 = model(text2["input_ids"].cuda(), attention_mask=text2["attention_mask"].cuda())
        loss = torch.nn.functional.mse_loss(min_max_normalization(hidden_states1[2][12]), min_max_normalization(hidden_states2[2][12]))
        s += torch.sqrt(loss).item()
        k += 1
    itrt2.set_postfix_str(f"rmse loss {(s/k):.5f}, num tokens {num_tokens}")