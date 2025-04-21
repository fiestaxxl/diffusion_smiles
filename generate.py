from diffusion import GaussianDiffusion
from model import DiffusionTransformerDecoder

import torch
import json

from utils import decode, process_decode
from tqdm import tqdm

with open("stoi.json", "r") as f:
    stoi = json.load(f)
itos = {v:k for k,v in stoi.items()}

timesteps = 2000

diffusion = GaussianDiffusion(timesteps,stoi, predict_xstart=True)
diffusion.initialize('quadratic')

emb_dim = 768
dim_model = 1024
time_dim = 1024
num_heads = 4
num_layers = 4
dim_ff = 4096
vocab_size = len(stoi)
bs = 128

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = DiffusionTransformerDecoder(emb_dim, time_dim, dim_model, num_heads, dim_ff, num_layers, vocab_size, pad_idx = stoi['<pad>'], dropout=0.2).to(device)


checkpoint = torch.load('checkpoint_new.pth')
model.load_state_dict(checkpoint['model'])
model.to(device)
model.eval()

generated_smiles = []

with torch.no_grad():
    for i in tqdm(range(15)):
        generated_embeddings = [model.get_logits(x["sample"]).softmax(dim=-1).argmax(dim=-1).cpu() for x in diffusion.p_sample_loop_progressive(model, (30, 109, emb_dim), time_steps=1500)]
        smiles = [[decode(process_decode(x.cpu().tolist(), [stoi['<sos>'],stoi['<eos>'],stoi['<pad>']]), itos) for x in y] for y in generated_embeddings[-2:]]
        generated_smiles.extend(smiles[-1])


import pickle

with open('smiles_generated.pickle', 'wb') as f:
    pickle.dump(generated_smiles, f)

with open('smiles_generated.pickle', 'rb') as f:
    gen = pickle.load(f)

# print('\n'.join(smiles[0]), '\n\n')
print('\n'.join(gen))






