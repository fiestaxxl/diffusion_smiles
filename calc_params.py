import os
import torch
from model import DiffusionTransformerDecoder

models = os.listdir()
models = [model for model in models if model.endswith('pth') and not 'checkpoint' in model]

emb_dim = 768
dim_model = 1024
time_dim = 1024
dim_ff = 4096
bs = 128


for model_name in models:
    params = model_name.split('_')
    num_layers = int(params[0])
    num_heads = int(params[2])
    model = DiffusionTransformerDecoder(emb_dim, time_dim, dim_model, num_heads, dim_ff, num_layers, 109, pad_idx = 108, dropout=0.2)

    pytorch_total_params = sum(p.numel() for p in model.parameters()) 
    print(f"Num_layers: {num_layers}, num_heads: {num_heads}, total_params: {pytorch_total_params}")  