from diffusion import GaussianDiffusion
from model import DiffusionTransformerDecoder
from train import TrainLoop, SMILESDataset
from torch.utils.data import DataLoader

import torch
import json
import pickle

from utils import encode


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
num_heads = 4#8
num_layers = 4#8
dim_ff = 4096
vocab_size = len(stoi)
bs = 128

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = DiffusionTransformerDecoder(emb_dim, time_dim, dim_model, num_heads, dim_ff, num_layers, vocab_size, 
                                    pad_idx = stoi['<pad>'], 
                                    dropout=0.2).to(device)


with open('../transformer_decoder/smiles.pickle', 'rb') as f:
    smiles = pickle.load(f)

max_len = len(smiles[-1])
print(max_len)

corrupt = finetune = False
dataset = SMILESDataset(smiles, encode, max_len, stoi, corrupt=corrupt)
dataloader = DataLoader(dataset, batch_size = bs, shuffle=True)

#10 epochs for 4 layer
#15 epochs for 12 layer
num_epochs = 1
max_iters = len(dataloader) * num_epochs


# checkpoint = torch.load('checkpoint_new.pth')
# model.load_state_dict(checkpoint['model'])

loop = TrainLoop(model = model, 
                 diffusion = diffusion, 
                 dataloader = dataloader, 
                 batch_size = bs, 
                 lr = 5e-5, 
                 max_iters = max_iters, 
                 weight_decay=0, 
                 gradient_clipping=3,
                 warmup=3500,
                 finetune=finetune,
                 
                #  use_scheduler=False,
                #  opt_params=checkpoint['optimizer']
                 )


loop.run_loop(num_epochs)





