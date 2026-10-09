"""Course formulas; R provides the exact rpart/ranger sampling and model defaults."""
import subprocess
import tempfile
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, brier_score_loss
FEATURES=['recency','frequency','monetary_value']
CATEGORIES=['home','sports','clothes','health','books','digital','toys']
SCHEMA={'customers':['user_id','last','electronics','nonelectronics',*CATEGORIES,'subscribe'],
 'products':['product_id','brand','RRP','screensize','technology','resolution','energy_class','support_HDR','refresh_rate'],
 'sales':['product_id','week_id','sales','discount'],
 'marketing':['brand','week_id','marketing_expense'],
 'experiments':['user_id','age','gender','account_age_days','pre_total_activity','treatment','post_total_activity']}

def demo_data(seed=42):
 r=np.random.default_rng(seed);n=2400
 categories={x:r.integers(0,6,n) for x in CATEGORIES}
 last=r.integers(1,365,n);frequency=sum(categories.values());money=np.round(frequency*r.uniform(30,120,n),2)
 p=1/(1+np.exp(-(-4-.012*last+.12*frequency+.0005*money)))
 customers=pd.DataFrame({'user_id':[f'DEMO-{i}' for i in range(n)],'last':last,'electronics':np.round(money*.6,2),'nonelectronics':np.round(money*.4,2),**categories,'subscribe':np.where(r.binomial(1,p),'yes','no')})
 rows=[]; brands=['Philips','Samsung','LG','Sony']
 for i in range(24):
  rows.append({'product_id':f'DEMO-P{i}','brand':brands[i//6],'RRP':float(350+i%6*110),
   'screensize':r.choice(['30–39','40–49','50–59','60+']),'technology':r.choice(['LCD','QLED','OLED']),
   'resolution':r.choice(['1080p','4K']),'energy_class':r.choice(['A','B','C']),
   'support_HDR':int(r.integers(0,2)),'refresh_rate':r.choice(['60','120'])})
 products=pd.DataFrame(rows);sales=[];marketing=[]
 for w in range(1,53):
  for bi,brand in enumerate(brands):
   spend=float(r.integers(600,2401));marketing.append({'brand':brand,'week_id':w,'marketing_expense':spend})
   for row in products[products.brand==brand].itertuples():
    disc=float(r.choice([0,.05,.10,.15,.20]));price=row.RRP*(1-disc)
    y=160-.12*price+.018*spend+bi*8+r.normal(0,10)
    sales.append({'product_id':row.product_id,'week_id':w,'sales':max(0,round(y)),'discount':disc})
 n=1800;group=np.repeat(['control','A','B'],600);pre=r.poisson(15,n)
 post=np.maximum(0,np.round(pre*.6+8+np.where(group=='A',6,np.where(group=='B',1,0))+r.normal(0,7,n)))
 exp=pd.DataFrame({'user_id':[f'DEMO-E{i}' for i in range(n)],'age':r.integers(18,65,n),'gender':r.choice(['F','M'],n),
  'account_age_days':r.integers(30,2500,n),'pre_total_activity':pre,'treatment':group,'post_total_activity':post})
 return {'customers':customers,'products':products,'sales':pd.DataFrame(sales),'marketing':pd.DataFrame(marketing),'experiments':exp}

def rfm(d):
 z=d.copy();z['recency']=z['last'];z['frequency']=z[CATEGORIES].sum(axis=1);z['monetary_value']=z.electronics+z.nonelectronics
 z['subscribe']=np.where(z.subscribe=='yes',1,0);return z

def validate(name,d):
 missing=set(SCHEMA[name])-set(d.columns)
 if missing:raise ValueError('Missing columns: '+', '.join(sorted(missing)))
 d=d[SCHEMA[name]].copy()
 if len(d)==0 or d.isna().any().any():raise ValueError('Empty data or missing values.')
 strings={'user_id','product_id','brand','screensize','technology','resolution','energy_class','refresh_rate','subscribe','gender','treatment'}
 for c in d.columns:
  if c not in strings:
   d[c]=pd.to_numeric(d[c],errors='raise')
   if not np.isfinite(d[c]).all() or (d[c]<0).any():raise ValueError(c+' must be finite and nonnegative.')
 if name=='customers':
  if len(d)<80 or d.user_id.duplicated().any():raise ValueError('At least 80 unique customer records required.')
  if set(d.subscribe.unique())!={'yes','no'} or d.subscribe.value_counts().min()<12:raise ValueError('subscribe must contain yes/no with at least 12 of each.')
 if name=='products':
  if d.product_id.duplicated().any() or (d.RRP<=0).any():raise ValueError('Unique products and positive RRP required.')
  if 'Philips' not in set(d.brand):raise ValueError('Q8 requires Philips as the reference brand.')
 if name=='sales' and (d.duplicated(['product_id','week_id']).any() or (d.discount>1).any()):raise ValueError('Duplicate product-week or discount outside 0–1.')
 if name=='marketing' and d.duplicated(['brand','week_id']).any():raise ValueError('Duplicate brand-week.')
 if name=='experiments':
  if d.user_id.duplicated().any() or not set(d.gender)<= {'F','M'}:raise ValueError('Unique users and gender coded F/M required.')
  if 'control' not in set(d.treatment) or d.treatment.nunique()<2:raise ValueError('Include control and treatment groups.')
  if d.groupby('treatment').size().min()<30:raise ValueError('At least 30 users per group required.')
 return d

def run_r(mode,d,output,*args):
 executable=shutil.which('Rscript')
 if executable is None:raise RuntimeError('Rscript is required for the original classroom models. Upload packages.txt and restart the deployment.')
 with tempfile.TemporaryDirectory() as temp:
  source=Path(temp)/'input.csv';d.to_csv(source,index=False)
  process=subprocess.run([executable,str(Path(__file__).with_name('course_backend.R')),mode,str(source),str(output),*map(str,args)],capture_output=True,text=True,timeout=240)
  if process.returncode:raise RuntimeError('Classroom R backend failed: '+process.stderr[-1500:])

class RModel:
 def __init__(self,model_bytes,d,prob):self.model_bytes=model_bytes;self.columns=d[FEATURES].copy();self.prob=np.array(prob)
 def predict_proba(self,x):
  if x.equals(self.columns):p=self.prob
  else:
   with tempfile.TemporaryDirectory() as temp:
    output=Path(temp)/'prediction.csv';model_path=Path(temp)/'model.rds';model_path.write_bytes(self.model_bytes);run_r('predict',x,output,model_path);p=pd.read_csv(output).probability.to_numpy()
  return np.column_stack([1-p,p])

def train_customers(d):
 with tempfile.TemporaryDirectory(prefix='mdl-classroom-') as temp:
  directory=Path(temp)
  run_r('train',d,directory)
  train=pd.read_csv(directory/'training.csv');test=pd.read_csv(directory/'test.csv')
  models={'Decision tree':RModel((directory/'tree.rds').read_bytes(),test,test.tree_probability),'Random forest':RModel((directory/'forest.rds').read_bytes(),test,test.forest_probability)}
  scores=[]
  for n,m in models.items():
   p=m.predict_proba(test[FEATURES])[:,1]
   scores.append({'Model':n,'Test AUC':roc_auc_score(test.subscribe,p),'Test Brier score':brier_score_loss(test.subscribe,p)})
  rules=(directory/'tree_rules.txt').read_text()
 return models,train,test,pd.DataFrame(scores),rules

def campaign(d,p,cost,margin,budget=None,all_customers=False):
 # Q1 uses the entire pilot. Q4/Q5 target ALL test cases strictly above break-even.
 eligible=np.arange(len(d)) if all_customers else np.flatnonzero(p>cost/margin)
 if budget is not None:
  eligible=eligible[np.argsort(-p[eligible],kind='stable')][:int(budget//cost)]
 responders=int(d.iloc[eligible].subscribe.sum());spent=len(eligible)*cost;profit=responders*margin-spent
 return {'Reached':len(eligible),'Responders':responders,'Spend':spent,'Net contribution':profit,'ROI':profit/spent if spent else np.nan}

def joined_data(products,sales,marketing):
 # Q7: products LEFT JOIN sales, then marketing RIGHT JOIN the result.
 d=products.merge(sales,on='product_id',how='left',validate='one_to_many')
 d=marketing.merge(d,on=['brand','week_id'],how='right',validate='one_to_many')
 if d.isna().any().any():raise ValueError('Q7 joins produced unmatched records; supply complete product/sales/marketing keys.')
 if set(sales.product_id)-set(products.product_id):raise ValueError('Sales includes unknown products.')
 d['final_price']=d.RRP*(1-d.discount)
 return d

def design(d,levels=None,extended=False):
 factors=['brand','screensize','technology','resolution','energy_class','support_HDR','refresh_rate'] if extended else ['brand']
 if levels is None:
  levels={c:sorted(d[c].astype(str).unique()) for c in factors}
  levels['brand']=['Philips']+[b for b in levels['brand'] if b!='Philips']
 columns={'Intercept':np.ones(len(d)),'final_price':d.final_price.to_numpy(),'marketing_expense':d.marketing_expense.to_numpy()}
 for c in factors:
  for v in levels[c][1:]:columns[c+'='+v]=(d[c].astype(str)==v).astype(float).to_numpy()
 return np.column_stack(list(columns.values())),list(columns),levels

def ols(X,y,names):
 rank=np.linalg.matrix_rank(X);df=len(y)-rank
 if df<=0:raise ValueError('Not enough rows for regression.')
 beta=np.linalg.lstsq(X,y,rcond=None)[0];variance=np.sum((y-X@beta)**2)/df;cov=variance*np.linalg.pinv(X.T@X)
 se=np.sqrt(np.maximum(0,np.diag(cov)));critical=stats.t.ppf(.975,df)
 p=2*stats.t.sf(np.divide(abs(beta),se,out=np.full_like(beta,np.inf),where=se>0),df)
 tab=pd.DataFrame({'Term':names,'Estimate':beta,'Std. error':se,'p-value':p,'95% lower':beta-critical*se,'95% upper':beta+critical*se})
 return beta,cov,df,tab

def fit_mix(d,extended=False):
 X,names,levels=design(d,extended=extended)
 if len(d)<40:raise ValueError('At least 40 product-week records required.')
 beta,cov,df,tab=ols(X,d.sales.to_numpy(),names)
 if extended:tab=tab[~tab.Term.str.startswith(('brand=','technology=','resolution=','energy_class=','support_HDR=','refresh_rate='))]
 return beta,cov,df,levels,tab

def experiment_results(d):
 control=d[d.treatment=='control'];out=[];balance=[]
 for name,g in d.groupby('treatment'):
  if name=='control':continue
  for c in ['age','gender','account_age_days','pre_total_activity']:
   a=(g[c]=='F').astype(int) if c=='gender' else g[c]
   b=(control[c]=='F').astype(int) if c=='gender' else control[c]
   t=stats.ttest_ind(a,b,equal_var=False)
   balance.append({'Comparison':name+' vs control','Variable':c,'Treatment mean':a.mean(),'Control mean':b.mean(),'Welch p':t.pvalue})
  a=g.post_total_activity;b=control.post_total_activity;t=stats.ttest_ind(a,b,equal_var=False)
  va=a.var(ddof=1)/len(a);vb=b.var(ddof=1)/len(b);se=np.sqrt(va+vb)
  df=(va+vb)**2/(va**2/(len(a)-1)+vb**2/(len(b)-1)) if se else np.inf
  ci=stats.t.ppf(.975,df)*se;diff=a.mean()-b.mean()
  out.append({'Variant':name,'N':len(g),'Mean post activity':a.mean(),'Effect vs control':diff,'95% lower':diff-ci,'95% upper':diff+ci,'Welch p':t.pvalue})
 groups=['control']+[g for g in sorted(d.treatment.unique()) if g!='control']
 X=np.column_stack([np.ones(len(d))]+[(d.treatment==g).astype(float) for g in groups[1:]])
 _,_,_,reg=ols(X,d.post_total_activity.to_numpy(),['Intercept (control mean)']+['Treatment '+g for g in groups[1:]])
 return pd.DataFrame(balance),pd.DataFrame(out),reg

def course_clv(N=5,membership=89,n_visit=40,revenue=100,profit_margin=.07,retention=.7,discount=.1,cpc=.4,click_trial=.1,trial_member=.2,promo=10,delivery=5):
 clicks=1/click_trial/trial_member;click_cost=cpc*clicks
 trial_profit=(revenue-promo)-revenue*(1-profit_margin)+revenue*profit_margin
 all_trial_profit=trial_profit/trial_member;delivery_cost=delivery*2/trial_member
 cac=click_cost+delivery_cost-all_trial_profit
 M=membership+revenue*n_visit*profit_margin;c=delivery*n_visit;g=M-c
 t=np.arange(1,N+1);flows=g*retention**(t-1)/(1+discount)**t
 return {'CAC':cac,'M':M,'c':c,'g':g,'Clicks per member':clicks,'Click cost':click_cost,'Trial contribution':all_trial_profit,'Trial delivery cost':delivery_cost,'CLV':float(flows.sum()-cac)},flows

def cluster_history(d,k,maximum=15,seed=123):
 with tempfile.TemporaryDirectory() as temp:
  output=Path(temp)/'history.csv';run_r('cluster',d,output,k,maximum,seed);return pd.read_csv(output)
