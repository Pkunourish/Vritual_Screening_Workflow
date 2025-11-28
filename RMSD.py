import spyrmsd
from spyrmsd import io, rmsd
from spyrmsd.molecule import Molecule
import numpy as np
from rdkit import Chem
import pandas as pd

# Process: Calculate RMSD between binding poses of vina and glide
# Sdf files containing binding poses from vina and glide are not provided and should be generated previously.
# Remember to remove all H atoms in both sdf files before running this code.
# Configuration
vina_suppl = Chem.SDMolSupplier('1443_vina.sdf', sanitize=False, removeHs=True)
glide_suppl = Chem.SDMolSupplier('1443_glide.sdf', removeHs=True)
output_file = "./rmsd_1443_select.csv"

# Map files contains all informations achieved in previous steps and will be read in every following processes.
df_map = pd.read_csv('./merge_1443.csv')

# Loading molecules
vina_mols = {}
suppl_vina = Chem.SDMolSupplier('1443_vina.sdf', sanitize=False, removeHs=True)
for mol in suppl_vina:
    if mol is not None:
        name = mol.GetProp('_Name')
        vina_mols[name] = mol
print(f"{len(vina_mols)} files in vina .sdf file")

glide_mols = {}
suppl_glide = Chem.SDMolSupplier('1443_glide.sdf', removeHs=True)
for mol in suppl_glide:
    if mol is not None:
        name = mol.GetProp('_Name')
        glide_mols[name] = mol
print(f"{len(glide_mols)} files in glide .sdf file")

# Matching molecules. RMSD calculation will only be carried out on matched molecules.
if len(vina_mols) != len(glide_mols):
    print("Warning: The number of molecules in Vina and Glide files do not match.")

matched_molecules = {}
missing_molecules = []

for index, row in df_map.iterrows():
    std_smiles = row['standardized_smiles']
    title_vina = row['title_vina']+'.pdbqt'
    title_glide = row['title_glide']
    
    if title_vina in vina_mols and title_glide in glide_mols:
        matched_molecules[std_smiles] = {
            'title_vina': title_vina,
            'title_glide': title_glide,
            'mol_vina': vina_mols[title_vina],
            'mol_glide': glide_mols[title_glide]
        }
    else:
        missing_info = []
        if title_vina not in vina_mols:
            missing_info.append(f"Vina: {title_vina}")
        if title_glide not in glide_mols:
            missing_info.append(f"Glide: {title_glide}")
        missing_molecules.append({
            'smiles': std_smiles,
            'missing': ", ".join(missing_info)
        })
    
    print(f"Matched molecules: {len(matched_molecules)}")

    if missing_molecules:
        print("Missing molecules:")
        for missing in missing_molecules:
            print(f"SMILES: {missing['smiles']}, Missing: {missing['missing']}")

# RMSD calculation with spyrmsd. spyrmsd would automatically align two molecules before calculation.
# Only heavy atoms are considered in RMSD calculation.
# It is ok if atoms are not listed in the same order, but different atom numbers will cause error.
# !!Important. Any H atoms should be removed previously as noted.
for cid in matched_molecules:
    ref = Molecule.from_rdkit(matched_molecules[cid]['mol_vina'])
    coords_ref = ref.coordinates
    anum_ref = ref.atomicnums
    adj_ref = ref.adjacency_matrix
    mol = Molecule.from_rdkit(matched_molecules[cid]['mol_glide'])
    coords = mol.coordinates
    anum = mol.atomicnums
    adj = mol.adjacency_matrix
    if coords_ref.shape[0] == coords.shape[0]:
        RMSD = rmsd.symmrmsd(coords_ref, coords, anum_ref, anum, adj_ref, adj)
        matched_molecules[cid]['RMSD'] = RMSD
    else:
        print(f"Unmatched atom count in {cid}: Vina({coords_ref.shape[0]}) vs Glide({coords.shape[0]})")

output_data = []
for cid, data in matched_molecules.items():
    output_data.append({
            'standardized_smiles': cid,
            'rmsd': data['RMSD']
        })
    
output_df = pd.DataFrame(output_data, columns = ['standardized_smiles','rmsd'])
output_df.sort_values(by = 'rmsd')

# RMSD results is added to the map file.
# A 2.5 Angstrom cutoff is applied. You can adjust it to 2.0 if you want a stricter screening.
df = output_df.merge(df_map, on='standardized_smiles', how='left')
rmsd_mol = df[df['rmsd']<2.51].copy()
rmsd_mol.to_csv(output_file, index = False)

# Optional: extract molecule names for binding pose extractions. dos format.
rmsd_mol['title_vina'].to_csv("./1443_rmsd_vinalist.txt", index = False, header = False)
rmsd_mol['title_glide'].to_csv("./1443_rmsd_glidelist.txt", index = False, header = False)