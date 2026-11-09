from pathlib import Path
import pandas as pd,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
P=Path(__file__).resolve().parents[1]; o=P/'results_full';m=pd.read_csv(o/'metrics.csv'); assert len(m)==81, f'Only {len(m)} of 81 expected model runs completed.'
g=m.groupby(['scenario','model','representation'])[['accuracy','macro_f1','balanced_accuracy']].agg(['mean','std']);g.to_csv(o/'summary_metrics.csv')
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':220})
fig,ax=plt.subplots(1,3,figsize=(8,3.6),sharey=True); models=['linear_svm','random_forest','xgboost']; labels=['SVM','Random forest','XGBoost']; colors=['#7182a5','#dfad4c','#247b7b']
for a,sc,title in zip(ax,['random','cold_s1','cold_s2'],['Random pair','One unseen drug','Two unseen drugs']):
 for i,(rp,label) in enumerate([('structure','Structure'),('target','Targets'),('structure_target','Structure + targets')]):
  vals=[g.loc[(sc,model,rp),('macro_f1','mean')] for model in models]; errs=[g.loc[(sc,model,rp),('macro_f1','std')] for model in models]
  a.bar(np.arange(3)+(i-1)*.24,vals,.24,yerr=errs,capsize=2,label=label,color=colors[i])
 a.set_title(title);a.set_xticks(range(3),labels,rotation=20,ha='right');a.set_ylim(0,1);a.grid(axis='y',alpha=.2)
ax[0].set_ylabel('Macro-F1 (mean ± SD)');handles,labels0=ax[0].get_legend_handles_labels();fig.legend(handles,labels0,loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(.5,1.10));fig.tight_layout();fig.savefig(o/'performance.png',bbox_inches='tight');plt.close(fig)
x=pd.read_csv(o/'analysis_data.csv'); fig,ax=plt.subplots(figsize=(7,3));x.type.value_counts().sort_index().plot.bar(ax=ax,color='#3c738d');ax.set(xlabel='Event class ID',ylabel='Drug pairs',title='Selected interaction-event sample');ax.tick_params(axis='x',rotation=0);fig.tight_layout();fig.savefig(o/'class_counts.png');plt.close(fig)
fig,ax=plt.subplots(figsize=(8,2)); ax.set_xlim(0,11);ax.set_ylim(0,2);ax.axis('off');
for i,(title,desc) in enumerate([('Public data','DrugBank-derived\nDDI + target table'),('Representations','Morgan fingerprints\n+ target indicators'),('Split design','Random pairs /\none or two new drugs'),('Evaluation','XGBoost + baselines\nMacro-F1, accuracy')]):
 x=.1+i*2.75;ax.add_patch(FancyBboxPatch((x,.4),2.25,1.1,boxstyle='round,pad=.08',fc='#e8f1f4',ec='#3c738d'));ax.text(x+1.12,1.18,title,ha='center',weight='bold',fontsize=11);ax.text(x+1.12,.72,desc,ha='center',fontsize=9)
 if i<3: ax.annotate('',xy=(x+2.65,.95),xytext=(x+2.38,.95),arrowprops=dict(arrowstyle='->',lw=1.4,color='#3c738d'))
fig.tight_layout();fig.savefig(o/'workflow.png');plt.close(fig)
print(g.round(4).to_string())
