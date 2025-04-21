from diffusion import GaussianDiffusion
from model import DiffusionTransformerDecoder
from train import TrainLoop, SMILESDataset
from torch.utils.data import DataLoader


import torch
import json
import pickle

from utils import encode, load_params


with open("stoi.json", "r") as f:
    stoi = json.load(f)
itos = {v:k for k,v in stoi.items()}

timesteps = 2000

#betas = get_named_beta_schedule('sqrt', timesteps)
diffusion = GaussianDiffusion(timesteps,stoi, predict_xstart=True)
diffusion.initialize('quadratic')

emb_dim = 768
dim_model = 1024
time_dim = 1024
num_heads = 8#8
num_layers = 9#8
dim_ff = 4096
vocab_size = len(stoi)
bs = 128

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = DiffusionTransformerDecoder(emb_dim, time_dim, dim_model, num_heads, dim_ff, num_layers, vocab_size, pad_idx = stoi['<pad>'], dropout=0.2).to(device)
old_state_dict = torch.load('8_layer_8_head.pth')['model']
old_state_params = list(old_state_dict.keys())


new_state_dict = load_params(model, old_state_dict)
model.load_state_dict(new_state_dict, strict=False) 

for name, param in model.named_parameters():
    if (name in old_state_params) and ('output_down_proj' not in name):
        param.requires_grad = False
    
    
for name, param in model.named_parameters():
    status = 'Frozen' if not param.requires_grad else "Trainable"
    print(f"{name}: {status}")



with open('../transformer_decoder/smiles.pickle', 'rb') as f:
    smiles = pickle.load(f)

max_len = len(smiles[-1])
print(max_len)

dataset = SMILESDataset(smiles, encode, max_len, stoi, corrupt=True, corrupt_ratio=0.2)
dataloader = DataLoader(dataset, batch_size = bs, shuffle=True)

num_epochs = 4
max_iters = len(dataloader) * num_epochs


loop = TrainLoop(model = model, 
                 diffusion = diffusion, 
                 dataloader = dataloader, 
                 batch_size = bs, 
                 lr = 5e-5, 
                 max_iters = max_iters, 
                 weight_decay=0, 
                 gradient_clipping=3,
                 warmup=400,
                 finetune=True,
                 tau=500)


loop.run_loop(num_epochs)





