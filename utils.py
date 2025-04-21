import torch

def encode(smiles, vocab):
    """Tokenizes a SMILES string based on the given vocabulary"""
    tokens = []
    i = 0
    length = len(smiles)

    while i < length:
        # Handle bracketed expressions like [C@H], [O-], etc.
        if smiles[i] == "[":
            j = i + 1
            while j < length and smiles[j] != "]":
                j += 1
            if j < length:  # Include the closing bracket
                tokens.append(smiles[i:j+1])
                i = j + 1
                continue

        # Handle two-character elements like Cl, Br
        if i + 1 < length and smiles[i:i+2] in vocab:
            tokens.append(smiles[i:i+2])
            i += 2
            continue

        # Handle single-character elements (C, N, O, etc.)
        if smiles[i] in vocab:
            tokens.append(smiles[i])

        i += 1  # Move to the next character

    # Convert tokens to indices
    token_ids = [vocab[token] for token in tokens if token in vocab]


    return token_ids

def decode(token_ids, vocab):
    if isinstance(token_ids, torch.Tensor):
      token_ids = token_ids.tolist()
    result = [vocab[id] for id in token_ids]
    return ''.join(result)

def process_decode(token_ids, special_ids, remove_special_tokens=True):
    sos_id, eos_id, pad_id = special_ids

    ids = []


    for id in token_ids:
        if id == eos_id or id == pad_id:
            break
        if remove_special_tokens and id == sos_id:
            continue
        ids.append(id)

    if not remove_special_tokens:
        ids.append(eos_id)
    return ids

def load_params(model, old_state_dict):
    new_state_dict = model.state_dict()
    old_state_params = list(old_state_dict.keys())

    l = len(old_state_params)
    i = 0

    while i<l:
        if old_state_params[i] in new_state_dict:
            new_state_dict[old_state_params[i]] = old_state_dict[old_state_params[i]]
        i+=1
    return new_state_dict