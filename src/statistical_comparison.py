"""Paired, exploratory model comparison using saved test scores.
Node-resampling ranges condition on trained models, splits, observed graph and class-capped sample.
Exact McNemar on drug-disjoint matched test pairs is conditional on independent matched outcomes.
"""
from pathlib import Path
import json,itertools
import numpy as np,pandas as pd,networkx as nx
from scipy.stats import binomtest
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,average_precision_score
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parents[1];R=P/'results_full'; df=pd.read_csv(R/'analysis_data.csv').set_index('sample_id');scores=pd.read_csv(R/'combined_test_scores.csv.gz');models=['xgboost','random_forest','linear_svm'];seeds=[11,22,33]; scenarios=['random','cold_s1','cold_s2']; contrasts=[('xgboost','random_forest'),('xgboost','linear_svm'),('random_forest','linear_svm')]; names={'xgboost':'XGBoost','random_forest':'Random forest','linear_svm':'SVM'}
all_drugs=sorted(set(df.d1)|set(df.d2)); idx={d:i for i,d in enumerate(all_drugs)}; groups={};metric_rows=[];match_rows=[]
for sc in scenarios:
 for seed in seeds:
  s=scores[(scores.scenario==sc)&(scores.seed==seed)]; piv={m:s[s.model==m].sort_values('sample_id').reset_index(drop=True) for m in models};ids=piv[models[0]].sample_id.to_numpy();truth=piv[models[0]]['true'].to_numpy();assert all(np.array_equal(ids,piv[m].sample_id.to_numpy()) and np.array_equal(truth,piv[m]['true'].to_numpy()) for m in models)
  info=df.loc[ids];i1=info.d1.map(idx).to_numpy();i2=info.d2.map(idx).to_numpy();pred={m:piv[m].pred.to_numpy() for m in models};groups[(sc,seed)]={'ids':ids,'truth':truth,'i1':i1,'i2':i2,'pred':pred}
  for m in models:
   c=confusion_matrix(truth,pred[m],labels=range(8)); tp=c.diagonal();tn=c.sum()-c.sum(axis=0)-c.sum(axis=1)+tp;fp=c.sum(axis=0)-tp;spec=float(np.mean(tn/(tn+fp)));arr=piv[m][[f'score_{i}' for i in range(8)]].to_numpy(); auc=float(np.mean([roc_auc_score(truth==k,arr[:,k]) for k in range(8)])); aupr=float(np.mean([average_precision_score(truth==k,arr[:,k]) for k in range(8)]));metric_rows.append({'scenario':sc,'seed':seed,'model':m,'n':len(ids),'accuracy':accuracy_score(truth,pred[m]),'macro_precision':precision_score(truth,pred[m],labels=range(8),average='macro',zero_division=0),'macro_recall':recall_score(truth,pred[m],labels=range(8),average='macro',zero_division=0),'macro_specificity':spec,'macro_f1':f1_score(truth,pred[m],labels=range(8),average='macro',zero_division=0),'macro_auroc_ovr':auc,'macro_auprc_ovr':aupr})
  graph=nx.Graph(); graph.add_edges_from(sorted(tuple(sorted((str(a),str(b)))) for a,b in zip(info.d1,info.d2))); matching=nx.max_weight_matching(graph,maxcardinality=True); selected={tuple(sorted(x)) for x in matching};mask=np.array([tuple(sorted((str(a),str(b)))) in selected for a,b in zip(info.d1,info.d2)]);assert mask.sum()==len(matching) and len(set(info.d1[mask])|set(info.d2[mask]))==2*mask.sum();
  for m1,m2 in contrasts:
   ok1=pred[m1][mask]==truth[mask];ok2=pred[m2][mask]==truth[mask];b1=int((ok1&~ok2).sum());c1=int((~ok1&ok2).sum());p=binomtest(b1,b1+c1,.5,alternative='two-sided').pvalue if b1+c1 else 1.; match_rows.append({'scenario':sc,'seed':seed,'model_a':m1,'model_b':m2,'matched_n':int(mask.sum()),'a_only_correct':b1,'b_only_correct':c1,'accuracy_difference':float(ok1.mean()-ok2.mean()),'p_raw':p,'sample_ids':' '.join(map(str,ids[mask]))})
metrics=pd.DataFrame(metric_rows);metrics.to_csv(R/'extended_metrics.csv',index=False);metrics.groupby(['scenario','model'])[['accuracy','macro_precision','macro_recall','macro_specificity','macro_f1','macro_auroc_ovr','macro_auprc_ovr']].agg(['mean','std']).to_csv(R/'extended_metrics_summary.csv')
matched=pd.DataFrame(match_rows); order=np.argsort(matched.p_raw.to_numpy());adjust=np.empty(len(matched));adjust[order]=np.maximum.accumulate(np.minimum(1.,matched.p_raw.to_numpy()[order]*(len(order)-np.arange(len(order)))));matched['p_holm_27']=adjust;matched.to_csv(R/'matched_mcnemar.csv',index=False)
def macrof(y,p,w):
 c=np.bincount(8*y+p,weights=w,minlength=64).reshape(8,8);actual=c.sum(axis=1)
 if np.any(actual==0):return np.nan
 denom=actual+c.sum(axis=0);return np.divide(2*np.diag(c),denom,out=np.zeros(8),where=denom>0).mean()
point={};
for sc in scenarios:
 for m in models: point[(sc,m)]=np.mean([macrof(groups[(sc,seed)]['truth'],groups[(sc,seed)]['pred'][m],np.ones(len(groups[(sc,seed)]['ids']))) for seed in seeds])
rng=np.random.default_rng(20261108);B=3000;boots={sc:[] for sc in scenarios}
for rep in range(B):
 count=rng.multinomial(len(all_drugs),np.full(len(all_drugs),1/len(all_drugs)))
 for sc in scenarios:
  vals={m:[] for m in models};valid=True
  for seed in seeds:
   q=groups[(sc,seed)];w=count[q['i1']]*count[q['i2']]
   for m in models:
    a=macrof(q['truth'],q['pred'][m],w)
    if np.isnan(a):valid=False
    vals[m].append(a)
  if valid:boots[sc].append({m:float(np.mean(vals[m])) for m in models})
rows=[]
for sc in scenarios:
 b=pd.DataFrame(boots[sc]);
 for m1,m2 in contrasts:
  diff=b[m1]-b[m2];lo,hi=np.quantile(diff,[.025,.975]);rows.append({'scenario':sc,'model_a':m1,'model_b':m2,'difference':point[(sc,m1)]-point[(sc,m2)],'node_range_low':lo,'node_range_high':hi,'valid_resamples':len(b),'total_resamples':B,'report_range':len(b)/B>=0.95})
paired=pd.DataFrame(rows);paired.to_csv(R/'paired_node_resampling.csv',index=False)
fig,axes=plt.subplots(1,3,figsize=(10,3.9),sharex=True,sharey=True)
for ax,sc,label in zip(axes,scenarios,['Random pair','One unseen drug','Two unseen drugs']):
 part=paired[paired.scenario==sc].reset_index(drop=True)
 for i,r in part.iterrows():

  if r.report_range: ax.plot([r.node_range_low,r.node_range_high],[i,i],color='#2f6b83',lw=2)
  ax.scatter([r.difference],[i],color='#b2533b',s=34,zorder=3)

 if sc=='cold_s2': ax.text(.04,.01,'Intervals suppressed*',transform=ax.transAxes,fontsize=8)
 ax.axvline(0,color='gray',ls='--',lw=1);ax.set_yticks(range(3),['XGBoost - RF','XGBoost - SVM','RF - SVM']);ax.set_title(label);ax.set_xlabel('Difference in macro-F1');ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
fig.tight_layout();fig.savefig(R/'paired_node_resampling.png',dpi=220,bbox_inches='tight');plt.close(fig)
print('MEAN METRICS');print(metrics.groupby(['scenario','model'])[['accuracy','macro_f1','macro_auroc_ovr']].agg(['mean','std']).round(4).to_string());print('NODE RANGES');print(paired.round(4).to_string(index=False));print('MCNEMAR HOLM');print(matched.drop(columns='sample_ids').round(4).to_string(index=False));print('Method caveat: node resampling is exploratory and conditional; matched-pair independence is unverified.')
