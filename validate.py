import rdkit
from rdkit import Chem
from rdkit.rdBase import BlockLogs
from rdkit.Chem import rdMolDescriptors, DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
import numpy as np
import Levenshtein
from scipy.spatial.distance import cosine
from fcd_torch import FCD
import numpy as np

# Load a sentence transformer for SMILES embeddings
# smiles_embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# def smiles_bleu(reference, hypotheses):
#     ref_embed = smiles_embedder.encode(reference)
#     hyp_embeds = smiles_embedder.encode(hypotheses)
#     return [1 - cosine(ref_embed, hyp) for hyp in hyp_embeds]

def levenshtein_distance(reference, hypotheses):
    return [Levenshtein.distance(reference, hyp) for hyp in hypotheses]

def tanimoto_similarity(fp1, fp2):
    block = BlockLogs()
    return DataStructs.TanimotoSimilarity(fp1, fp2)

def compute_fingerprint_metrics(reference, hypotheses):
    block = BlockLogs()
    mol_ref = Chem.MolFromSmiles(reference)
    if not mol_ref:
        return None, None, None
    
    maccs_ref = rdMolDescriptors.GetMACCSKeysFingerprint(mol_ref)
    rdk_ref = FingerprintMols.FingerprintMol(mol_ref)
    morgan_ref = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol_ref, 2, nBits=1024)
    
    results = []
    for hypothesis in hypotheses:
        mol_hyp = Chem.MolFromSmiles(hypothesis)
        if not mol_hyp:
            results.append((None, None, None))
            continue
        
        maccs_hyp = rdMolDescriptors.GetMACCSKeysFingerprint(mol_hyp)
        rdk_hyp = FingerprintMols.FingerprintMol(mol_hyp)
        morgan_hyp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol_hyp, 2, nBits=1024)
        
        results.append((
            tanimoto_similarity(maccs_ref, maccs_hyp),
            tanimoto_similarity(rdk_ref, rdk_hyp),
            tanimoto_similarity(morgan_ref, morgan_hyp),
        ))
    return zip(*results)

def exact_match(reference, hypotheses):
    return [int(reference == hyp) for hyp in hypotheses]

def validity(hypotheses):
    block = BlockLogs()
    return [int(Chem.MolFromSmiles(hyp) is not None) for hyp in hypotheses]

def get_valid_mols(hypotheses):
    block = BlockLogs()
    return [hyp for hyp in hypotheses if (Chem.MolFromSmiles(hyp) is not None)]

def compute_fcd_score(references, hypotheses):
    return FCD(references, hypotheses)

def evaluate_smiles_metrics(reference_smiles, generated_smiles):
    #bleu_scores = smiles_bleu(reference_smiles, generated_smiles)
    lev_distances = levenshtein_distance(reference_smiles, generated_smiles)
    maccs, rdk, morgan = compute_fingerprint_metrics(reference_smiles, generated_smiles)
    exact_matches = exact_match(reference_smiles, generated_smiles)
    #validities = validity(generated_smiles)
    
    return [{
        #"SMILES BLEU": bleu,
        "Levenshtein Distance": lev,
        "MACCS FTS": mac,
        "RDK FTS": rd,
        "Morgan FTS": morg,
        "Exact Match": ex,
        #"Validity": val
    } for #bleu, 
            lev, mac, rd, morg, ex, 
            #val 
            in zip(
        #bleu_scores, 
        lev_distances, maccs, rdk, morgan, exact_matches,
         # validities
          )]

# Example Usage
# reference = "CCO"
# generated = ["CCN", "CCC", "CCO"]
# metrics = evaluate_smiles_metrics(reference, generated)
# print(metrics)


def evaluate_model_quality(reference_smiles_list, generated_smiles_list):
    all_metrics = []
    
    for ref_smiles in reference_smiles_list:
        metrics = evaluate_smiles_metrics(ref_smiles, generated_smiles_list)
        all_metrics.append(metrics)

    # Compute average for each metric
    avg_metrics = {key: round(float(np.mean([m[key] for batch in all_metrics for m in batch if m[key] is not None])),3)
                   for key in all_metrics[0][0]}
    
    return avg_metrics

# Example usage
# reference_set = ["CCO", "CNC", "CCC"]  # Example real molecules
# generated_set = ["CCN", "CCO", "CNC", "CCCl"]  # Generated molecules

# model_quality = evaluate_model_quality(reference_set, generated_set)
# print(model_quality)

if __name__=='__main__':
    import pickle
    with open('data/smiles.pickle', 'rb') as f:
        smiles = np.array(pickle.load(f))

    with open('smiles_generated.pickle', 'rb') as f:
        generated_smiles = np.array(pickle.load(f))

    
    smiles_len = len(generated_smiles)

    valid_mols = get_valid_mols(generated_smiles)
    reference_mols = np.random.choice(smiles, len(valid_mols), replace=False)
    model_quality = evaluate_model_quality(reference_mols, valid_mols[:10])
    model_quality['Validity'] = round(len(valid_mols)/smiles_len,3)
    print(model_quality)
    print('\nValidMols:\n', '\n'.join(valid_mols))
    

'''
{'Levenshtein Distance': 31.641, 'MACCS FTS': 0.308, 'RDK FTS': 0.342, 'Morgan FTS': 0.109, 'Exact Match': 0.0, 'Validity': 0.085} - finetuned - 9 layers, 8 heads, t=200, 12min
{'Levenshtein Distance': 32.941, 'MACCS FTS': 0.257, 'RDK FTS': 0.361, 'Morgan FTS': 0.097, 'Exact Match': 0.0, 'Validity': 0.093} - 8 layers, 8 heads, t=200, 12min
{'Levenshtein Distance': 33.0, 'MACCS FTS': 0.267, 'RDK FTS': 0.381, 'Morgan FTS': 0.094, 'Exact Match': 0.0, 'Validity': 0.038} - 4 layers, 4 heads, t=200, 7min
{'Levenshtein Distance': 33.277, 'MACCS FTS': 0.301, 'RDK FTS': 0.358, 'Morgan FTS': 0.101, 'Exact Match': 0.0, 'Validity': 0.0415} - 4 layers, 4 heads, full_loss, t=200, 7 min
{'Levenshtein Distance': 37.702, 'MACCS FTS': 0.146, 'RDK FTS': 0.317, 'Morgan FTS': 0.042, 'Exact Match': 0.0, 'Validity': 0.207} - 4 layers, 4 heads, full_loss, with full special tokens mask, t=200
{'Levenshtein Distance': 32.167, 'MACCS FTS': 0.313, 'RDK FTS': 0.34, 'Morgan FTS': 0.098, 'Exact Match': 0.0, 'Validity': 0.043} - 4 layers, 4 heads, full_loss, with 0.2* addition to special tokens pos att_scores
{'Levenshtein Distance': 35.154, 'MACCS FTS': 0.204, 'RDK FTS': 0.365, 'Morgan FTS': 0.089, 'Exact Match': 0.0, 'Validity': 0.062} - 4 layers, 4 heads, full_loss, with f(t)* addition to special tokens pos att_scores. Model learns not to predict () at all
{'Levenshtein Distance': 32.855, 'MACCS FTS': 0.342, 'RDK FTS': 0.304, 'Morgan FTS': 0.1, 'Exact Match': 0.0, 'Validity': 0.024} - 4 layers, 4 heads, full_loss, 30 epochs
{'Levenshtein Distance': 31.357, 'MACCS FTS': 0.309, 'RDK FTS': 0.378, 'Morgan FTS': 0.113, 'Exact Match': 0.0, 'Validity': 0.103 - 12 layers, 8 heads, t=200, 20 min}
'''