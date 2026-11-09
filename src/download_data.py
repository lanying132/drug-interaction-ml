from pathlib import Path
import hashlib,requests,zipfile
P=Path(__file__).resolve().parents[1]/'data';P.mkdir(exist_ok=True)
files={'Deng_dataset.zip':('https://raw.githubusercontent.com/cgwcbz2/CSK-DDI/main/Deng%27s_dataset.zip','8812674ac543e05a2d5a2400f67537ca8645d2a1f7f5cfe2e1aa28ae91c27381'),'drug_target_9606.csv':('https://raw.githubusercontent.com/hanjunwei-lab/DsimSynergy/main/Code/Drug%20Sim%20Network/Data/drug_target_9606.csv','25c625502b8d8ff9acf0f4415b720f2a0b7c866975d829667b67d3e6fb9e3e77')}
for name,(url,sha) in files.items():
 r=requests.get(url,timeout=120);r.raise_for_status();b=r.content
 if hashlib.sha256(b).hexdigest()!=sha: raise ValueError('Source changed; expected hash differs: '+name)
 (P/name).write_bytes(b);print(name,len(b),sha)
with zipfile.ZipFile(P/'Deng_dataset.zip') as z:
 for entry in z.infolist():
  target=(P/entry.filename).resolve()
  if P.resolve() not in target.parents and target!=P.resolve(): raise ValueError('Unsafe zip entry')
 z.extractall(P)

r=requests.get('https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi',params={'db':'pubmed','id':'42949104,32407508,29666228,31802250,26446141,34740542,42236581','retmode':'xml'},timeout=60);r.raise_for_status();(P/'pubmed_references.xml').write_bytes(r.content);print('Retrieved reference metadata from PubMed')

r=requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",params={"db":"pubmed","id":"40588546,38177736,42477148,41611854,35817846,41729823,41697921,38113076,37903412,37401373,37195815,42906047,42670852,42478169,42260929,40399143,40310752,38132638,42949104,41091924","retmode":"xml"},timeout=60);r.raise_for_status();(P/"selected_pubmed.xml").write_bytes(r.content);print("Retrieved selected reference metadata from PubMed")
