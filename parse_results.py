import re
import ast
from collections import defaultdict

from utils import encode
import json
import math

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

def parse_smiles_file(filepath):
    results = defaultdict(list)
    with open(filepath, 'r') as f:
        content = f.read()

    # Split by section
    sections = re.split(r'=== PARAMS: checkpoint=checkpoint_(\d+)\.pth timesteps=(\d+) ===', content)

    # The first split section is before the first PARAMS block; ignore it
    for i in range(1, len(sections), 3):
        checkpoint = int(sections[i])
        timesteps = int(sections[i+1])
        block = sections[i+2].strip()

        # Extract metrics dict
        metrics_match = re.search(r"\{.*?\}", block, re.DOTALL)
        metrics = ast.literal_eval(metrics_match.group()) if metrics_match else {}

        # Extract SMILES lines under "ValidMols:"
        smiles = []
        if "ValidMols:" in block:
            smiles_block = block.split("ValidMols:")[1].strip()
            smiles = [line.strip() for line in smiles_block.splitlines() if line.strip()]

        results[checkpoint].append({'timestep': timesteps, 'smiles':smiles, 'metrics': metrics})
        # results.append({
        #     "checkpoint": checkpoint,
        #     "timesteps": timesteps,
        #     "smiles": smiles
        # })

    return results

def smiles_complexity(smiles, max_len=109, vocab_size = 78):
    with open("stoi.json", "r") as f:
        vocab = json.load(f)

    token_ids = encode(smiles, vocab)

    length = len(token_ids)
    num_unique_tokens = len(set(token_ids))

    return (length/max_len * num_unique_tokens/vocab_size), length, num_unique_tokens


def plot_validity_subplots(results):
    num_checkpoints = len(results)
    # Arrange subplots in a grid, roughly square
    cols = math.ceil(math.sqrt(num_checkpoints))
    rows = math.ceil(num_checkpoints / cols)

    fig = make_subplots(rows=rows, cols=cols,
                        subplot_titles=[f'Checkpoint {ckpt}' for ckpt in results.keys()])

    # Mapping from checkpoint idx to subplot (row, col)
    checkpoint_list = list(results.keys())

    for idx, checkpoint in enumerate(checkpoint_list):
        row = idx // cols + 1
        col = idx % cols + 1

        data = results[checkpoint]
        timesteps = []
        validities = []

        for batch in data:
            timestep = batch['timestep']
            timesteps.append(timestep)
            validity = batch['metrics'].get('Validity', None)
            validities.append(validity*100)

        if timesteps and validities:
            timesteps, validities = zip(*sorted(zip(timesteps, validities)))
            fig.add_trace(
                go.Scatter(
                    x=timesteps,
                    y=validities,
                    mode='lines+markers',
                    name=f'Checkpoint {checkpoint}',
                    showlegend=False  # Hide duplicate legends
                ),
                row=row, col=col
            )

    fig.update_layout(
        height=300 * rows,
        width=350 * cols,
        template='plotly_white'
    )

    fig.update_xaxes(title_text='Timestep')
    fig.update_yaxes(title_text='Validity, %')

    fig.show()
    fig.write_image("test.png", scale=2) 


def plot_all_validity_curves_on_one_plot(results):
    fig = go.Figure()

    for checkpoint, data in results.items():
        timesteps = []
        validities = []

        for batch in data:
            timestep = batch['timestep']
            timesteps.append(timestep)
            validity = batch['metrics'].get('Validity', None)
            validities.append(validity*100)

        if timesteps and validities:
            timesteps, validities = zip(*sorted(zip(timesteps, validities)))
            fig.add_trace(go.Scatter(
                x=timesteps,
                y=validities,
                mode='lines+markers',
                name=f'{checkpoint*20} training steps'
            ))

    fig.update_layout(
        xaxis_title='Timestep',
        yaxis_title='Validity, %',
        template='plotly_white',
        legend_title='Checkpoints',
        width=900,
        height=600
    )

    fig.show()

    fig.write_image("validity_all_checkpoints.png", scale=2)


def plot_all_metrics(results):

    cols = 3
    rows = 1

    fig = make_subplots(rows=rows, cols=cols,
                        subplot_titles=['Complexity', 'Average Smiles Length', "Average Unique Tokens"])
    #use_checkpoints = [10000, 30000, 90000]
    for checkpoint, data in results.items():
        timesteps = []
        complexities = []
        lengths = []
        tokens = []
        #if checkpoint in use_checkpoints:
        for batch in data:
            timestep = batch['timestep']
            timesteps.append(timestep)
            smiles = batch['smiles']

            len_smiles = len(smiles)+1e-5

            metrics = [smiles_complexity(smile) for smile in smiles]
            complexity = sum([metric[0] for metric in metrics])/len_smiles
            smiles_len = sum([metric[1] for metric in metrics])/len_smiles
            unique_tokens = sum([metric[2] for metric in metrics])/len_smiles

            complexities.append(complexity)
            lengths.append(smiles_len)
            tokens.append(unique_tokens)

        timesteps, complexities, lengths, tokens = zip(*sorted(zip(timesteps, complexities, lengths, tokens)))
        # max_comp = max(complexities)
        # complexities = [comp/max_comp for comp in complexities]
        # timesteps, lengths = zip(*sorted(zip(timesteps, lengths)))
        # timesteps, tokens = zip(*sorted(zip(timesteps, tokens)))
        
        fig.add_trace(
            go.Scatter(
                x=timesteps,
                y=complexities,
                mode='lines+markers',
                name=f'{checkpoint*20} training steps' 
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=timesteps,
                y=lengths,
                mode='lines+markers',
                name=f'{checkpoint*20} training steps'
            ),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(
                x=timesteps,
                y=tokens,
                mode='lines+markers',
                name=f'{checkpoint*20} training steps',
            ),
            row=1, col=3
        )


    fig.update_layout(
        height=500 * rows,
        width=550 * cols,
        #title_text='Metrics vs Timesteps for Each Checkpoint',
        template='plotly_white',
        legend_title='Checkpoints',
    )

    fig.update_xaxes(title_text='Timestep')

    fig.show()
    fig.write_image("all_metrics.png", scale=2)
#print(smiles_complexity('C'))

if __name__=='__main__':
    parsed_data = parse_smiles_file("outputs/all_results_4_layers.txt")
    # Example usage
    #plot_validity_subplots(parsed_data)
    # print('calc validity')
    plot_all_validity_curves_on_one_plot(parsed_data)
    # print('calc metrics')
    plot_all_metrics(parsed_data)
    # print(parsed_data[10000])

    print(smiles_complexity('C'))
    print(smiles_complexity('CCCCCC'))
    print(smiles_complexity('CFCClCCCBr'))
    print(smiles_complexity('Cc1ccc(C(=O)N[C@@H]2CCC[C@H]2[NH+]2CC[C@@H]2Cc2c(Br)cc(I)cc2C)cc1'))

