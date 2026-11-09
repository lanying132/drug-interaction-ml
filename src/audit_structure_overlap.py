"""Audit canonical SMILES overlap between identifiers in the sampled data."""
from pathlib import Path
import json,pandas as pd,numpy as np
from rdkit import Chem,RDLogger
RDLogger.DisableLog('rdApp.*');P=Path(__file__).resolve().parents[1];df=pd.read_csv(P/'results_full/analysis_data.csv');s=pd.read_csv(P/"data/Deng's_dataset/drug_smiles.csv").drop_duplicates('drug_id').set_index('drug_id').smiles.to_dict();drugs=sorted(set(df.d1)|set(df.d2));cm={d:Chem.MolToSmiles(Chem.MolFromSmiles(s[d]),isomericSmiles=True) for d in drugs};out={'sample_drug_ids':len(drugs),'canonical_isomeric_smiles':len(set(cm.values())),'cold_splits':[]}
for seed in [11,22,33]:
 rng=np.random.default_rng(seed);unseen=set(rng.choice(drugs,size=round(.2*len(drugs)),replace=False));tr=df[~df.d1.isin(unseen)&~df.d2.isin(unseen)];seen=set(tr.d1)|set(tr.d2);train_smiles={cm[d] for d in seen};out['cold_splits'].append({'seed':seed,'heldout_drug_ids':len(unseen),'heldout_with_canonical_smiles_identical_to_training':sum(cm[d] in train_smiles for d in unseen)})
(P/'results_full/structure_overlap_audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))
