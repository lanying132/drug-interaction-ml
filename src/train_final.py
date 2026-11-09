"""Fit the prespecified combined-feature XGBoost model to all sampled records.
This artifact is for reproducibility and demonstration, not independent evaluation.
"""
from pathlib import Path
import json,numpy as np,pandas as pd,joblib
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from rdkit import Chem,DataStructs,RDLogger
from rdkit.Chem import AllChem
from scipy.sparse import csr_matrix,hstack
from xgboost import XGBClassifier
RDLogger.DisableLog('rdApp.*');P=Path(__file__).resolve().parents[1]; D=P/'data';O=P/'model';O.mkdir(exist_ok=True)
df=pd.read_csv(P/'results_full/analysis_data.csv'); sm=pd.read_csv(D/"Deng's_dataset"/'drug_smiles.csv').drop_duplicates('drug_id').set_index('drug_id').smiles.to_dict();gen=AllChem.GetMorganGenerator(radius=2,fpSize=512); fps={}
for d,s in sm.items():
 m=Chem.MolFromSmiles(str(s))
 if m:
  a=np.zeros(512,dtype=np.uint8);DataStructs.ConvertToNumpyArray(gen.GetFingerprint(m),a);fps[d]=a
t=pd.read_csv(D/'drug_target_9606.csv').dropna(subset=['drugbank_id','uniprot_id']);t=t[t.drugbank_id.isin(fps)];genes=sorted(set(t.uniprot_id));gm={g:i for i,g in enumerate(genes)};tm={d:set(v.uniprot_id) for d,v in t.groupby('drugbank_id')};drugs=sorted(set(df.d1)|set(df.d2)); ix={d:i for i,d in enumerate(drugs)};f=np.stack([fps[d] for d in drugs]); tg=np.zeros((len(drugs),len(genes)),dtype=np.uint8)
for i,d in enumerate(drugs):
 for g in tm.get(d,[]):tg[i,gm[g]]=1
a=df.d1.map(ix).to_numpy();b=df.d2.map(ix).to_numpy(); X=hstack([csr_matrix(f[a]),csr_matrix(f[b]),csr_matrix(tg[a]),csr_matrix(tg[b])],format='csr')
model=XGBClassifier(n_estimators=100,max_depth=5,learning_rate=.1,subsample=.9,colsample_bytree=.9,objective='multi:softprob',num_class=8,tree_method='hist',n_jobs=4,random_state=8122);model.fit(X,df.type);model.save_model(O/'xgboost_combined.json'); svm=LinearSVC(C=1,random_state=8122,max_iter=3000); svm.fit(X,df.type); joblib.dump(svm,O/'linear_svm_combined.joblib',compress=3); rf=RandomForestClassifier(n_estimators=100,min_samples_leaf=2,max_features='sqrt',n_jobs=4,random_state=8122); rf.fit(X,df.type); joblib.dump(rf,O/'random_forest_combined.joblib',compress=3); (O/'feature_schema.json').write_text(json.dumps({'fingerprint_bits':512,'radius':2,'proteins':genes,'trained_rows':len(df),'random_seed':8122,'classes':list(range(8))},indent=2));print('Saved three final all-data models; these fits are not independent validation.')
