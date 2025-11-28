import pandas as pd
from rdkit import Chem

# Process: Merge glide and vina results for finding common molecules.
# Configuration
vina_df = pd.read_csv("sort_1443_vinascore_top10.csv")
glide_df = pd.read_csv("sort_1443_glidescore_top10.csv")

def standarize_smiles(smiles):
    # Convert SMILES to standardized canonical form
    #!!Important. Vina results contain no stereochemistry while glide results contain.
    mol = Chem.MolFromSmiles(smiles)
    standardized_smiles = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
    return standardized_smiles

# Find common molecules based on standardized SMILES
vina_df['standardized_smiles'] = vina_df['SMILES'].apply(standarize_smiles)
glide_df['standardized_smiles'] = glide_df['SMILES'].apply(standarize_smiles)
glide_df.rename(columns={
    "SMILES": "SMILES",
    "title": "title",
    "r_i_glide_gscore": "score"
}, inplace=True)
common_smiles = set(vina_df["standardized_smiles"]).intersection(glide_df["standardized_smiles"])
common_vina = vina_df[vina_df['standardized_smiles'].isin(common_smiles)]
common_glide = glide_df[glide_df['standardized_smiles'].isin(common_smiles)]

# Merge the two dataframes on standardized SMILES
merged_df = pd.merge(
    common_vina[["title", "SMILES", "standardized_smiles", "score"]],
    common_glide[["title", "SMILES", "standardized_smiles", "score"]],
    on="standardized_smiles",
    suffixes=("_vina", "_glide")
)
merged_df.rename(columns={
    "SMILES_vina": "Original_SMILES_vina",
    "SMILES_glide": "Original_SMILES_glide"
}, inplace=True)
merged_df.to_csv('./merge_1443.csv', index=False)

# Optional: extract molecule names for binding pose extractions. dos format.
merged_df['title_vina'].to_csv('./1443_vinalist.txt', index=False)
merged_df['title_glide'].to_csv('./1443_glidelist.txt', index=False)