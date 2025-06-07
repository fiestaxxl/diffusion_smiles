import re
import ast
from collections import defaultdict

from utils import encode
import json
import math

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

def smiles_complexity(smiles, max_len=109, vocab_size = 78):
    with open("stoi.json", "r") as f:
        vocab = json.load(f)

    token_ids = encode(smiles, vocab)

    length = len(token_ids)
    num_unique_tokens = len(set(token_ids))

    return (length/max_len * num_unique_tokens/vocab_size), length, num_unique_tokens

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

def is_dominated(p1, p2, keys):
    return all(p2[k] <= p1[k] for k in keys) and any(p2[k] < p1[k] for k in keys)

def pareto_front(data, keys):
    front = []
    for p in data:
        if not any(is_dominated(p, q, keys) for q in data if q != p):
            front.append(p)
    return front


if __name__=='__main__':
    results = parse_smiles_file("experiments/8_layers_6000_timesteps/all_results_8_layers_6000_timesteps.txt")
    transformed_data = []

    preprocessed_data = []
    for d in results[70000]:
        if d['metrics'].get('Validity'):
            len_smiles = len(d['smiles'])+1e-5
            metrics = [smiles_complexity(smile) for smile in d['smiles']]
            complexity = sum([metric[0] for metric in metrics])/len_smiles
            smiles_len = sum([metric[1] for metric in metrics])/len_smiles
            unique_tokens = sum([metric[2] for metric in metrics])/len_smiles


            preprocessed_data.append({
                'validity': d['metrics']['Validity'],
                'complexity': complexity,
                'num_tokens': unique_tokens,
                'length': smiles_len,
                'num_steps': d['timestep'],  # already a minimization metric
            })


    # Compute min and max for each metric
    min_max = {}
    metrics = ['num_steps', 'validity', 'complexity', 'num_tokens', 'length']
    for m in metrics:
        values = [d[m] for d in preprocessed_data]
        min_val = min(values)
        max_val = max(values)
        min_max[m] = (min_val, max_val)

    # Normalize a single data point
    def normalize_point(d, min_max):
        return {m: (d[m] - min_max[m][0]) / (min_max[m][1] - min_max[m][0]) if min_max[m][1] != min_max[m][0] else 0.0 for m in metrics}

    # Create normalized dataset
    normalized_data = [normalize_point(d, min_max) for d in preprocessed_data]

    for d in normalized_data:

        transformed_data.append({
            'neg_validity': -d['validity'],
            'neg_complexity': -d['complexity'],
            'neg_num_tokens': -d['num_tokens'],
            'neg_length': -d['length'],
            'num_steps': d['num_steps'],  # already a minimization metric
            'original': d,  # keep original data for later retrieval
        })

    metrics = ['num_steps', 'neg_validity', 'neg_complexity', 'neg_num_tokens', 'neg_length']
    front = pareto_front(transformed_data, metrics)
    pareto_points = [p['original'] for p in front]

    # All points
    m1, m2 = 'length', 'complexity'

    # import plotly.express as px
    # import plotly.graph_objects as go

    # # Prepare "All" points
    # x_all = [d[m1] for d in preprocessed_data]
    # y_all = [d[m2] for d in preprocessed_data]

    # # Prepare Pareto front, sorted by num_steps
    # pareto_sorted = sorted(pareto_points, key=lambda p: p[m1])
    # x_pareto = [p[m1] for p in pareto_sorted]
    # y_pareto = [p[m2] for p in pareto_sorted]

    # # Create figure
    # fig = go.Figure()

    # # Scatter: all points
    # fig.add_trace(go.Scatter(
    #     x=x_all,
    #     y=y_all,
    #     mode='markers',
    #     name='All',
    #     opacity=0.3,
    #     marker=dict(color='gray')
    # ))

    # # Pareto front: scatter + line
    # fig.add_trace(go.Scatter(
    #     x=x_pareto,
    #     y=y_pareto,
    #     mode='markers+lines',
    #     name='Pareto Front',
    #     marker=dict(color='red', size=8),
    #     line=dict(color='red', width=2)
    # ))

    # # Layout
    # fig.update_layout(
    #     title=f'Pareto Front: {m1} vs {m2}',
    #     xaxis_title=f'{m1}',
    #     yaxis_title=f'{m2}',
    #     legend_title='Legend',
    #     template='plotly_white'
    # )

    #fig.show()
    #fig.write_image("sample_plot.png", scale=2)


    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Define metrics
    metrics = ['num_steps', 'validity', 'complexity', 'num_tokens', 'length']


    # Create subplot grid
    n = len(metrics)
    fig = make_subplots(rows=n, cols=n, 
                        shared_xaxes='all', shared_yaxes='all',
                        horizontal_spacing=0.02, vertical_spacing=0.02)

    # Fill in subplots
    for i, y_metric in enumerate(metrics):
        for j, x_metric in enumerate(metrics):
            # if i == j:
            #     continue  # skip diagonal

            # All data
            x_all = [d[x_metric] for d in normalized_data]
            y_all = [d[y_metric] for d in normalized_data]

            # Pareto
            pareto_sorted = sorted(pareto_points, key=lambda p: p[x_metric])
            x_pareto = [p[x_metric] for p in pareto_sorted]
            y_pareto = [p[y_metric] for p in pareto_sorted]

            # Add all points
            fig.add_trace(go.Scatter(
                x=x_all,
                y=y_all,
                mode='markers',
                marker=dict(color='gray', size=4),
                showlegend=False
            ), row=i+1, col=j+1)

            # Add pareto points
            fig.add_trace(go.Scatter(
                x=x_pareto,
                y=y_pareto,
                mode='lines+markers',
                marker=dict(color='red', size=6),
                showlegend=False
            ), row=i+1, col=j+1)

    # Update axis titles
    for i, metric in enumerate(metrics):
        fig.update_yaxes(title_text=metric, row=i+1, col=1)
        fig.update_xaxes(title_text=metric, row=n, col=i+1)

    fig.update_layout(
        height=250 * n,
        width=250 * n,
        title='Scatter Plot Matrix with Pareto Front Highlighted: 140000_6000_steps',
        showlegend=False,
        template='plotly_white'
    )

    fig.write_image("140000_6000_steps.png", scale=2)



