import pandas as pd
import re
import json

df = pd.read_csv('../transformer_decoder/250k_rndm_zinc_drugs_clean_3.csv')

smiles = df.smiles.to_list()
smiles = list(map(lambda x: x[:-1], smiles))
smiles_text = ' '.join(smiles)
un_char = sorted(set("".join(smiles_text)))+['<sos>']  + ['<eos>']  + ['<pad>']

periodic_elements = ['H', 'He', 'Li', 'Be', 'B', 'C', "N", 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb',
                     'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn']

result = dict()
result_small = dict()

for el in periodic_elements:
  result[el] = el in smiles_text
  result_small[el.lower()] = el.lower() in smiles_text


elements_big = [key for key, value in result.items() if value]
elements_low = [key for key, value in result_small.items() if value]

elements_big.extend(elements_low)

elements_big = elements_big[:-5]

matches = set(re.findall(r"\[[^\]]*\]", smiles_text))

symbols = elements_big
elements_not_here = ['Li', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', '[N--]', '[N@@+]', '[N@H+]', '[CH]']
numbers = [str(i) for i in range(1,10)]
chars = ['(', ')', '/', '\\', '=', '-', '+', '#'] + ['<sos>']  + ['<eos>']  + ['<pad>']
symbols.extend(elements_not_here)
symbols.extend(matches)
symbols.extend(numbers)
symbols.extend(chars)

stoi = { ch:i for i,ch in enumerate(symbols) }
itos = { i:ch for i,ch in enumerate(symbols) }

if __name__ == '__main__':
    with open("stoi.json", "w") as f:
        json.dump(stoi, f)