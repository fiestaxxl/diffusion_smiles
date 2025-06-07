from diffusion import GaussianDiffusion
from model import DiffusionTransformerDecoder
from train import TrainLoop, SMILESDataset
from torch.utils.data import DataLoader

import torch
import json
import pickle

from utils import encode

import wandb

with open("stoi.json", "r") as f:
    stoi = json.load(f)
itos = {v:k for k,v in stoi.items()}

with open('data/smiles.pickle', 'rb') as f:
    smiles = pickle.load(f)

max_len = len(smiles[-1])
print(max_len)


config = {
    'timesteps': 6000,
    'emb_dim': 768,
    'dim_model': 1024,
    'time_dim': 1024,
    'num_heads': 8,#8
    'num_layers': 8,#8
    'dim_ff': 4096,
    'dropout': 0.1,

    'vocab_size': len(stoi),
    'max_len': max_len,
    'bs': 172,
    'betas_shape': 'quadratic',

    'corrupt': True,
    'num_epochs': 50,
    'lr': 5e-5,

    
}


#betas = get_named_beta_schedule('sqrt', timesteps)
diffusion = GaussianDiffusion(config['timesteps'],
                                stoi, 
                                predict_xstart=True)

diffusion.initialize(config['betas_shape'])


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = DiffusionTransformerDecoder(emb_dim=config['emb_dim'], 
                                    time_dim=config['time_dim'], 
                                    dim_model=config['dim_model'], 
                                    num_heads=config['num_heads'], 
                                    dim_ff=config['dim_ff'], 
                                    num_layers=config['num_layers'], 
                                    vocab_size=config['vocab_size'], 
                                    pad_idx = stoi['<pad>'], 
                                    dropout=config['dropout']).to(device)


corrupt = finetune = False
dataset = SMILESDataset(smiles, encode, max_len, stoi, corrupt=True)
dataloader = DataLoader(dataset, batch_size = config['bs'], shuffle=True)

#10 epochs for 4 layer
#15 epochs for 12 layer
num_epochs = config['num_epochs']
max_iters = len(dataloader) * num_epochs

config['max_iters'] = max_iters
config['warmup'] = int(config['max_iters']*0.05)

# checkpoint = torch.load('checkpoint_new.pth')
# model.load_state_dict(checkpoint['model'])

loop = TrainLoop(model = model, 
                 diffusion = diffusion, 
                 dataloader = dataloader, 
                 batch_size = config['bs'], 
                 lr = config['lr'], 
                 max_iters = max_iters, 
                 weight_decay=0, 
                 gradient_clipping=3,
                 warmup=config['warmup'],
                #  use_scheduler=False,
                #  opt_params=checkpoint['optimizer']
                 )

wandb.init(
    project = "DiffusionSmilesTransformer",
    config = config,
)

loop.run_loop(num_epochs)





