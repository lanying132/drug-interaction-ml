from pathlib import Path
import os,json,hashlib,sys,time
os.environ['OMP_NUM_THREADS']='4'; os.environ['OPENBLAS_NUM_THREADS']='4'
import numpy as np, pandas as pd
from scipy.sparse import csr_matrix,hstack,vstack
from rdkit import Chem,DataStructs,RDLogger
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,f1_score,balanced_accuracy_score,confusion_matrix
from xgboost import XGBClassifier
RDLogger.DisableLog('rdApp.*')
ROOT=Path(__file__).resolve().parents[1]; DAT=ROOT/'data'; OUT=ROOT/'results_full'; OUT.mkdir(exist_ok=True)
p=DAT/"Deng's_dataset"; allrows=pd.concat([pd.read_csv(p/'0'/f'ddi_{s}1xiao.csv') for s in ['training','validation','test']],ignore_index=True)
allrows['pair']=allrows.apply(lambda z:'|'.join(sorted((z.d1,z.d2))),axis=1)
assert not allrows.pair.duplicated().any()
sm=pd.read_csv(p/'drug_smiles.csv').drop_duplicates('drug_id').set_index('drug_id').smiles.to_dict()
fpgen=AllChem.GetMorganGenerator(radius=2,fpSize=512)
fps={}
for d,s in sm.items():
 m=Chem.MolFromSmiles(str(s))
 if m is not None:
  a=np.zeros(512,dtype=np.uint8); DataStructs.ConvertToNumpyArray(fpgen.GetFingerprint(m),a); fps[d]=a
valid=allrows[allrows.d1.isin(fps)&allrows.d2.isin(fps)&allrows.type.isin(range(8))].copy()
parts=[g.sample(n=min(len(g),1200),random_state=8122) for _,g in valid.groupby('type')]; df=pd.concat(parts).sample(frac=1,random_state=8122).reset_index(drop=True); df['sample_id']=np.arange(len(df)); df.to_csv(OUT/'analysis_data.csv',index=False)
target=pd.read_csv(DAT/'drug_target_9606.csv').dropna(subset=['drugbank_id','uniprot_id']); target=target[target.drugbank_id.isin(fps)]
drugs=sorted(set(df.d1)|set(df.d2)); genes=sorted(set(target.uniprot_id)); gene_index={g:i for i,g in enumerate(genes)}
tmap={d:set(v.uniprot_id) for d,v in target.groupby('drugbank_id')}; feat=np.zeros((len(drugs),512),np.uint8); tg=np.zeros((len(drugs),len(genes)),np.uint8)
for i,d in enumerate(drugs):
 feat[i]=fps[d]
 for g in tmap.get(d,[]): tg[i,gene_index[g]]=1
idx={d:i for i,d in enumerate(drugs)}; i1=df.d1.map(idx).to_numpy(); i2=df.d2.map(idx).to_numpy();
Xchem=hstack([csr_matrix(feat[i1]),csr_matrix(feat[i2])],format='csr'); Xtarget=hstack([csr_matrix(tg[i1]),csr_matrix(tg[i2])],format='csr'); Xboth=hstack([Xchem,Xtarget],format='csr'); y=df.type.to_numpy()
metadata={'source_rows':len(allrows),'source_classes':int(allrows.type.nunique()),'valid_top8_rows':len(valid),'sample_rows':len(df),'sample_class_counts':{str(k):int(v) for k,v in df.type.value_counts().sort_index().items()},'unique_drugs':len(drugs),'mapped_target_drugs':sum(d in tmap for d in drugs),'target_unique_proteins':len(genes),'sample_seed':8122,'fingerprint_radius':2,'fingerprint_bits':512,'max_per_class':1200,'target_missing_encoded_zero':True,'input_sha256':{f:hashlib.sha256((DAT/f).read_bytes()).hexdigest() for f in ['Deng_dataset.zip','drug_target_9606.csv']}}
(OUT/'data_summary.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8'); print(json.dumps(metadata,indent=2),flush=True)
records=[]; predictions=[]; split_manifest=[]
for seed in [11,22,33]:
 tr,te=train_test_split(np.arange(len(df)),test_size=.2,random_state=seed,stratify=y)
 rng=np.random.default_rng(seed); testdrugs=set(rng.choice(drugs,size=round(.2*len(drugs)),replace=False))
 unseen1=df.d1.isin(testdrugs).to_numpy(); unseen2=df.d2.isin(testdrugs).to_numpy(); coldtrain=np.flatnonzero(~unseen1 & ~unseen2); seen_drugs=set(df.iloc[coldtrain].d1)|set(df.iloc[coldtrain].d2); observed1=df.d1.isin(seen_drugs).to_numpy(); observed2=df.d2.isin(seen_drugs).to_numpy(); s1=np.flatnonzero((unseen1 & observed2) | (unseen2 & observed1)); s2=np.flatnonzero(unseen1 & unseen2)
 assert not(set(df.iloc[coldtrain].d1)|set(df.iloc[coldtrain].d2)) & testdrugs
 for scenario,train,test in [('random',tr,te),('cold_s1',coldtrain,s1),('cold_s2',coldtrain,s2)]:
  assert not set(df.iloc[train].pair)&set(df.iloc[test].pair)
  split_manifest.append({'seed':seed,'scenario':scenario,'train':len(train),'test':len(test),'train_classes':int(len(set(y[train]))),'test_classes':int(len(set(y[test]))),'test_unseen_drugs':len((set(df.iloc[test].d1)|set(df.iloc[test].d2))-(set(df.iloc[train].d1)|set(df.iloc[train].d2)))})
  for rep,X in [('structure',Xchem),('target',Xtarget),('structure_target',Xboth)]:
   for name in ['linear_svm','random_forest','xgboost']:
    if name=='linear_svm': model=LinearSVC(C=1,random_state=seed,max_iter=3000)
    elif name=='random_forest': model=RandomForestClassifier(n_estimators=100,min_samples_leaf=2,max_features='sqrt',n_jobs=4,random_state=seed)
    else: model=XGBClassifier(n_estimators=100,max_depth=5,learning_rate=.1,subsample=.9,colsample_bytree=.9,objective='multi:softprob',num_class=8,tree_method='hist',n_jobs=4,random_state=seed)
    start=time.time();model.fit(X[train],y[train]); pred=model.predict(X[test]); metrics={'seed':seed,'scenario':scenario,'representation':rep,'model':name,'n_train':len(train),'n_test':len(test),'accuracy':accuracy_score(y[test],pred),'macro_f1':f1_score(y[test],pred,labels=list(range(8)),average='macro',zero_division=0),'balanced_accuracy':balanced_accuracy_score(y[test],pred),'seconds':time.time()-start}; records.append(metrics)
    predictions.extend({'seed':seed,'scenario':scenario,'representation':rep,'model':name,'sample_id':int(df.sample_id.iloc[j]),'true':int(y[j]),'pred':int(k)} for j,k in zip(test,pred)); pd.DataFrame(records).to_csv(OUT/'metrics.csv',index=False); print(json.dumps(metrics),flush=True)
(OUT/'splits.json').write_text(json.dumps(split_manifest,indent=2),encoding='utf-8');pd.DataFrame(predictions).to_csv(OUT/'predictions.csv.gz',index=False,compression='gzip')
