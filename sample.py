from diffusion import GaussianDiffusion
from model import DiffusionTransformerDecoder

import torch
import json

import argparse

from utils import decode, process_decode
from tqdm import tqdm
from validate import get_valid_mols
from parse_results import smiles_complexity
from collections import defaultdict

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


with open("stoi.json", "r") as f:
    stoi = json.load(f)
itos = {v:k for k,v in stoi.items()}

timesteps = 1000

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

    ts = [10, 20, 40, 60, 80, 100, 120, 150]
    eta = torch.linspace(0, 0.50, 5)
    results = defaultdict(list)
    N = 3

    for e in eta:
        validity, complexity, smiles_len, unique_tokens = 0, 0, 0, 0

        for _ in range(N):
            for i in tqdm(range(100)):
                generated_embeddings = [model.get_logits(x["sample"]).softmax(dim=-1).argmax(dim=-1).cpu() for x in diffusion.ddim_sample_loop(model, (5, 109, emb_dim), time_steps=50, eta=e)]
                smiles = [[decode(process_decode(x.cpu().tolist(), [stoi['<sos>'],stoi['<eos>'],stoi['<pad>']]), itos) for x in y] for y in generated_embeddings[-2:]]
                generated_smiles.extend(smiles[-1])

        
            valids = get_valid_mols(generated_smiles)
            len_valids = len(valids)

            metrics = [smiles_complexity(smile) for smile in valids]
            validity += len_valids/len(generated_smiles)*100
            complexity += sum([metric[0] for metric in metrics])/len_valids
            smiles_len += sum([metric[1] for metric in metrics])/len_valids
            unique_tokens += sum([metric[2] for metric in metrics])/len_valids

        results['validity'].append(validity/N)
        results['complexity'].append(complexity/N)
        results['length'].append(smiles_len/N)
        results['tokens'].append(unique_tokens/N)

        # print(f'Validity: {len_valids/len(generated_smiles)*100}%\n')
        # print(f'Complexity: {complexity}')
        # print(f'Average length: {smiles_len}')
        # print(f'Unique tokens: {unique_tokens}')
        # print('\n'.join(valids))


    # Create subplots: 1 column, n rows
    fig = make_subplots(rows=1, cols=4, shared_xaxes=True, subplot_titles=list(results.keys()))

    # Add traces to each subplot
    i = 1
    for key, value in results.items():
        fig.add_trace(
            go.Scatter(x=eta, 
                    y=value,
                    mode='lines+markers',
                    name=key,
                    ),
            row=1,
            col=i
        )
        i += 1


    # Update layout
    fig.update_xaxes(title_text='Timestep')
    fig.update_layout(height=1200, width=1000, title_text="f(x) for Multiple Series (Subplots)")
    fig.show()
    fig.write_image("gen_images/results.png", scale=2) 



