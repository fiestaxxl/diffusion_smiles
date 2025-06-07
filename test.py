import pickle, json
from utils import encode, decode, process_decode
from validate import get_valid_mols
# with open('smiles_generated.pickle', 'rb') as f:
#     gen = pickle.load(f)
# with open("stoi.json", "r") as f:
#     stoi = json.load(f)

# itos = {v:k for k,v in stoi.items()}

# for step in gen[::20]:
#     print(decode(process_decode(encode(step[-1], stoi), [stoi['<sos>'],stoi['<eos>'],stoi['<pad>']]), itos))

print(get_valid_mols(['CCCCN1C(=Occ2nnnnc2111cccccc)ccccccccccccccccccccccc']))