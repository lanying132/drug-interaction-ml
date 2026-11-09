from pathlib import Path
import pandas as pd,numpy as np,json
from sklearn.metrics import accuracy_score,f1_score,balanced_accuracy_score
p=Path(__file__).resolve().parents[1]/'results_full';m=pd.read_csv(p/'metrics.csv');pred=pd.read_csv(p/'predictions.csv.gz');data=pd.read_csv(p/'analysis_data.csv').set_index('sample_id');keys=['seed','scenario','model','representation'];assert len(m)==81 and len(m.groupby(keys))==81
for row in m.itertuples(index=False):
 t=pred[(pred.seed==row.seed)&(pred.scenario==row.scenario)&(pred.model==row.model)&(pred.representation==row.representation)]; assert len(t)==row.n_test and not t.sample_id.duplicated().any();assert np.array_equal(t['true'].to_numpy(),data.loc[t.sample_id,'type'].to_numpy());assert abs(f1_score(t['true'],t.pred,labels=range(8),average='macro',zero_division=0)-row.macro_f1)<1e-12;assert abs(accuracy_score(t['true'],t.pred)-row.accuracy)<1e-12
for (seed,scenario),part in pred.groupby(['seed','scenario']):
 sets=[set(g.sample_id) for _,g in part.groupby(['model','representation'])];assert all(s==sets[0] for s in sets)
print('PASS: 81 model runs, saved predictions match reported metrics and source labels; all models and features share the same test examples within each split.')
(p/'verification.txt').write_text('PASS: 81 model runs, predictions match metrics and source labels; test examples shared across models and features within each split.\n')
