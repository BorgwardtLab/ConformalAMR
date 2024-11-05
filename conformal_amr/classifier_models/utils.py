import re
from urllib.parse import quote
from urllib.request import urlopen

import matplotlib.pyplot as plt
import networkx as nx
from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem.SaltRemover import SaltRemover


def mol_to_nx(mol):
    G = nx.Graph()

    for atom in mol.GetAtoms():
        G.add_node(
            atom.GetIdx(),
            atomic_num=atom.GetAtomicNum(),
            is_aromatic=atom.GetIsAromatic(),
            atom_symbol=atom.GetSymbol(),
        )

    for bond in mol.GetBonds():
        G.add_edge(
            bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond_type=bond.GetBondType()
        )

    return G


def create_ab_graph(drug_name):
    identifiers = re.split(r"(?<!\d)-", drug_name)
    smiles = []
    smiles_dict = {
        "Ceftarolin": "CCO/N=C(/C1=NSC(=N1)N)\C(=O)N[C@H]2[C@@H]3N(C2=O)C(=C(CS3)SC4=NC(=CS4)C5=CC=[N+](C=C5)C)C(=O)[O-]",
    }

    # Retrieve the SMILES for each drug
    for drug in identifiers:
        try:
            ans = smiles_dict[drug]
        except KeyError:
            url = (
                "http://cactus.nci.nih.gov/chemical/structure/"
                + quote(drug)
                + "/smiles"
            )
            ans = urlopen(url).read().decode("utf8")

        smiles.append(ans)

    # Create the molecule from the SMILES
    mols = [Chem.MolFromSmiles(smile) for smile in smiles]

    # Remove water from the molecule
    water = Chem.MolFromSmiles("[OH2]")
    mols = [rdmolops.DeleteSubstructs(mol, water, onlyFrags=True) for mol in mols]

    # Remove salts from the molecule
    remover = SaltRemover(defnData="[Cl,Br,Na,K,Ca]")
    mols = [remover.StripMol(mol) for mol in mols]

    # Create the graph from the molecule
    graphs = [mol_to_nx(mol) for mol in mols]

    if len(graphs) > 1:
        for i in range(1, len(graphs)):
            graphs[0] = nx.disjoint_union(graphs[0], graphs[i])

    return graphs[0], smiles, mols


def calc_conv1d_output_size(input_size, kernel_sizes, stride=1, padding=0, dilation=1):
    size = input_size
    for ks in kernel_sizes:
        size = (size + 2 * padding - dilation * (ks - 1) - 1) / stride + 1
    return size
