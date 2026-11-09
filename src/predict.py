"""Return exploratory event-ID probabilities for an ordered SMILES/DrugBank pair."""
from pathlib import Path
import argparse,json,numpy as np,pandas as pd,joblib
from rdkit import Chem,DataStructs
from rdkit.Chem import AllChem
from scipy.sparse import csr_matrix,hstack
from xgboost import XGBClassifier
P=Path(__file__).resolve().parents[1];parser=argparse.ArgumentParser();parser.add_argument('--model',choices=['xgboost','linear_svm','random_forest'],default='xgboost');parser.add_argument('--smiles1',required=True);parser.add_argument('--smiles2',required=True);parser.add_argument('--drugbank1',default='');parser.add_argument('--drugbank2',default='');args=parser.parse_args();schema=json.loads((P/'model/feature_schema.json').read_text());ix={x:i for i,x in enumerate(schema['proteins'])};gen=AllChem.GetMorganGenerator(radius=schema['radius'],fpSize=schema['fingerprint_bits']); t=pd.read_csv(P/'data/drug_target_9606.csv').dropna(subset=['drugbank_id','uniprot_id'])
def feature(smiles,db):
 mol=Chem.MolFromSmiles(smiles)
 if mol is None:raise ValueError('Invalid SMILES')
 fp=np.zeros(schema['fingerprint_bits'],dtype=np.uint8);DataStructs.ConvertToNumpyArray(gen.GetFingerprint(mol),fp);v=np.zeros(len(ix),dtype=np.uint8)
 for g in t.loc[t.drugbank_id==db,'uniprot_id']:
  if g in ix:v[ix[g]]=1
 return fp,v
f1,t1=feature(args.smiles1,args.drugbank1);f2,t2=feature(args.smiles2,args.drugbank2);X=hstack([csr_matrix(v.reshape(1,-1)) for v in (f1,f2,t1,t2)],format='csr')
if args.model=='xgboost':
 model=XGBClassifier();model.load_model(P/'model/xgboost_combined.json')
else: model=joblib.load(P/f'model/{args.model}_combined.joblib')
if args.model=='linear_svm':
 values=model.decision_function(X)[0]; score_type='uncalibrated_decision_scores'
else:
 values=model.predict_proba(X)[0]; score_type='class_probabilities_not_calibrated'
print(json.dumps({'model':args.model,'event_id':int(model.predict(X)[0]),'score_type':score_type,'scores':{str(i):float(p) for i,p in zip(model.classes_,values)},'target1_known':bool(t1.any()),'target2_known':bool(t2.any()),'note':'Eight-class research output for known interacting pairs; not a test of whether interaction occurs or a clinical safety assessment.'},indent=2))
