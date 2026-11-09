"""Re-fit the combined input models to recover per-class test scores.
Assert exact agreement with previously recorded test predictions.
"""
from pathlib import Path
import os,json
os.environ['OMP_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='4'
import numpy as np,pandas as pd
from rdkit import Chem,DataStructs,RDLogger
from rdkit.Chem import AllChem
from scipy.sparse import csr_matrix,hstack
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
RDLogger.DisableLog('rdApp.*');P=Path(__file__).resolve().parents[1];D=P/'data';R=P/'results_full'; df=pd.read_csv(R/'analysis_data.csv');old=pd.read_csv(R/'predictions.csv.gz');y=df.type.to_numpy();sm=pd.read_csv(D/"Deng's_dataset"/'drug_smiles.csv').drop_duplicates('drug_id').set_index('drug_id').smiles.to_dict(); gen=AllChem.GetMorganGenerator(radius=2,fpSize=512);fps={}
for d,s in sm.items():
 m=Chem.MolFromSmiles(str(s))
 if m is not None:
  v=np.zeros(512,dtype=np.uint8);DataStructs.ConvertToNumpyArray(gen.GetFingerprint(m),v);fps[d]=v
target=pd.read_csv(D/'drug_target_9606.csv').dropna(subset=['drugbank_id','uniprot_id']);target=target[target.drugbank_id.isin(fps)];genes=sorted(set(target.uniprot_id));gi={g:i for i,g in enumerate(genes)}; tmap={d:set(g.uniprot_id) for d,g in target.groupby('drugbank_id')};drugs=sorted(set(df.d1)|set(df.d2));idx={d:i for i,d in enumerate(drugs)};a=np.stack([fps[d] for d in drugs]); b=np.zeros((len(drugs),len(genes)),np.uint8)
for i,d in enumerate(drugs):
 for g in tmap.get(d,[]):b[i,gi[g]]=1
ix=df.d1.map(idx).to_numpy();iy=df.d2.map(idx).to_numpy();X=hstack([csr_matrix(a[ix]),csr_matrix(a[iy]),csr_matrix(b[ix]),csr_matrix(b[iy])],format='csr')
records=[]
for seed in [11,22,33]:
 tr,te=train_test_split(np.arange(len(df)),test_size=.2,random_state=seed,stratify=y);rng=np.random.default_rng(seed);td=set(rng.choice(drugs,size=round(.2*len(drugs)),replace=False));u1=df.d1.isin(td).to_numpy();u2=df.d2.isin(td).to_numpy();ct=np.flatnonzero(~u1 & ~u2);seen=set(df.iloc[ct].d1)|set(df.iloc[ct].d2);s1=np.flatnonzero((u1 & df.d2.isin(seen).to_numpy()) | (u2 & df.d1.isin(seen).to_numpy()));s2=np.flatnonzero(u1 & u2)
 for sc,train,test in [('random',tr,te),('cold_s1',ct,s1),('cold_s2',ct,s2)]:
  for name in ['linear_svm','random_forest','xgboost']:
   if name=='linear_svm': model=LinearSVC(C=1,random_state=seed,max_iter=3000)
   elif name=='random_forest':model=RandomForestClassifier(n_estimators=100,min_samples_leaf=2,max_features='sqrt',n_jobs=4,random_state=seed)
   else:model=XGBClassifier(n_estimators=100,max_depth=5,learning_rate=.1,subsample=.9,colsample_bytree=.9,objective='multi:softprob',num_class=8,tree_method='hist',n_jobs=4,random_state=seed)
   model.fit(X[train],y[train]);pred=model.predict(X[test]); scores=model.decision_function(X[test]) if name=='linear_svm' else model.predict_proba(X[test]);prev=old[(old.seed==seed)&(old.scenario==sc)&(old.model==name)&(old.representation=='structure_target')].set_index('sample_id').loc[test]; assert np.array_equal(prev.pred.to_numpy(),pred),f'Predictions changed: {seed} {sc} {name}';assert np.array_equal(model.classes_,np.arange(8))
   for row,j in enumerate(test): records.append({'seed':seed,'scenario':sc,'model':name,'sample_id':int(j),'true':int(y[j]),'pred':int(pred[row]),**{f'score_{c}':float(scores[row,c]) for c in range(8)}})
   print(seed,sc,name,len(test),flush=True)
pd.DataFrame(records).to_csv(R/'combined_test_scores.csv.gz',index=False,compression='gzip');print('Saved scores; all 27 predicted-label sets match earlier experiments.',flush=True)
