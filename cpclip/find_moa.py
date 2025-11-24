from chembl_webresource_client.new_client import new_client
from rdkit import Chem
import pandas as pd
from tqdm import tqdm

def fetch_moa_from_smiles(df: pd.DataFrame, smiles_col: str = "SMILES", name_col: str = "compound") -> pd.DataFrame:
    moa_data = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Fetching MOA"):
        smiles = row[smiles_col]
        name = row[name_col]

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        results = new_client.similarity.filter(smiles=smiles, similarity=100)
        if not results:
            continue

        chembl_id = results[0]['molecule_chembl_id']
        moas = new_client.mechanism.filter(molecule_chembl_id=chembl_id)

        for m in moas:
            moa_data.append({
                'compound': name,
                'target_name': m.get('target_pref_name', ''),
                'mechanism_of_action': m.get('mechanism_of_action', ''),
                'action_type': m.get('action_type', '')
            })

    return pd.DataFrame(moa_data)

csv_path = r"xxx"
output_csv_path = r"xxx"

compound_df = pd.read_csv(csv_path)
moa_df = fetch_moa_from_smiles(compound_df)
moa_df.to_csv(output_csv_path, index=False)