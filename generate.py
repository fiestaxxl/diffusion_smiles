from diffusion import GaussianDiffusion
from model import DiffusionTransformerDecoder

import torch
import json

import argparse

from utils import decode, process_decode
from tqdm import tqdm

with open("stoi.json", "r") as f:
    stoi = json.load(f)
itos = {v:k for k,v in stoi.items()}

timesteps = 2000

diffusion = GaussianDiffusion(timesteps,stoi, predict_xstart=True)
diffusion.initialize('quadratic')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Generate smiles.pickle with given parameters.")
    parser.add_argument('--num_heads', type=int, required=False, default=8, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, required=False, default=8, help='Number of encoder layers')
    parser.add_argument('--path_checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--timesteps', type=int, required=True, help='Number of diffusion timesteps')
    
    args = parser.parse_args()
    
    print(f"Generating with num_heads={args.num_heads}, num_layers={args.num_layers}, model={args.path_checkpoint} and timesteps={args.timesteps}")
    
    emb_dim = 768
    dim_model = 1024
    time_dim = 1024
    num_heads = args.num_heads
    num_layers = args.num_layers
    dim_ff = 4096
    vocab_size = len(stoi)
    bs = 128

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = DiffusionTransformerDecoder(emb_dim, time_dim, dim_model, num_heads, dim_ff, num_layers, vocab_size, pad_idx = stoi['<pad>'], dropout=0.2).to(device)

    path_checkpoint = args.path_checkpoint # 'checkpoints/checkpoint_90000.pth'
    checkpoint = torch.load(path_checkpoint, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    generated_smiles = []


    for i in tqdm(range(10)):
        generated_embeddings = [model.get_logits(x["sample"]).softmax(dim=-1).argmax(dim=-1).cpu() for x in diffusion.p_sample_loop_progressive(model, (30, 109, emb_dim), time_steps=args.timesteps)]
        smiles = [[decode(process_decode(x.cpu().tolist(), [stoi['<sos>'],stoi['<eos>'],stoi['<pad>']]), itos) for x in y] for y in generated_embeddings[-2:]]
        generated_smiles.extend(smiles[-1])


    import pickle

    with open('smiles_generated.pickle', 'wb') as f:
        pickle.dump(generated_smiles, f)

    with open('smiles_generated.pickle', 'rb') as f:
        gen = pickle.load(f)

    # print('\n'.join(smiles[0]), '\n\n')
    #print('\n'.join(gen))






