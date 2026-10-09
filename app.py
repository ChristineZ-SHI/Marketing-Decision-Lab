import io
import zipfile
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.calibration import calibration_curve
from sklearn.tree import export_text
from analytics import (FEATURES, SCHEMA, demo_data, validate, train_customers,
    campaign, joined_data, fit_mix, design, experiment_results, rfm, course_clv, cluster_history)

st.set_page_config(page_title='Marketing Decision Lab', page_icon='◉', layout='wide')
# Use a fresh template without trace defaults to avoid shared-template mutation.
px.defaults.template=go.layout.Template(layout=go.Layout(
    paper_bgcolor='white', plot_bgcolor='white', font=dict(color='#20251B'),
    xaxis=dict(gridcolor='#E8ECE2', zerolinecolor='#DEE3D7'),
    yaxis=dict(gridcolor='#E8ECE2', zerolinecolor='#DEE3D7')))
px.defaults.color_discrete_sequence=['#93BA3F','#B8D67B','#658B2D','#485E30','#9FAA90']
st.markdown('''<style>
.stApp {background:#F6F7F3;color:#20251B}.block-container{max-width:1500px;padding-top:6rem}
[data-testid="stHeader"]{background:#F6F7F3!important;color:#20251B!important;border-bottom:1px solid #DEE3D7}
[data-testid="stHeader"] button,[data-testid="stHeader"] button *,
[data-testid="stHeader"] a{color:#20251B!important}
[data-testid="stHeader"] svg{color:#20251B!important;fill:currentColor}
h1,h2,h3{color:#171B14;letter-spacing:-.025em}
[data-testid="stSidebar"]{background:#0B0D0B;border-right:1px solid #2D3527}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p{color:#F4F6F0}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"]{color:#C0C6B9}
[data-testid="stMetric"]{background:white;border:1px solid #DEE3D7;border-top:3px solid #93BA3F;border-radius:10px;padding:1.1rem}
[data-testid="stMetricLabel"],[data-testid="stCaptionContainer"]{color:#526048}
[data-testid="stMetricValue"]{color:#171B14}
[data-testid="stMain"] [data-testid="stPlotlyChart"]{background:white;border:1px solid #DEE3D7;border-radius:10px;padding:.6rem}
[data-testid="stMain"] [data-testid="stMarkdownContainer"]{color:#20251B}
[data-testid="stMain"] input{color:#20251B;background:white}
[data-testid="stButton"] button,[data-testid="stDownloadButton"] button{border-radius:6px}
[data-testid="stMain"] [data-testid="stButton"] button,
[data-testid="stMain"] [data-testid="stDownloadButton"] button{
background:#FFFFFF!important;color:#111111!important;border:1px solid #111111!important}
[data-testid="stMain"] [data-testid="stButton"] button *,
[data-testid="stMain"] [data-testid="stDownloadButton"] button *{color:#111111!important}
[data-testid="stMain"] [data-testid="stButton"] button:hover,
[data-testid="stMain"] [data-testid="stDownloadButton"] button:hover{background:#EEEEEE!important}
.eyebrow{color:#658B2D;font-weight:700;letter-spacing:.16em;font-size:.8rem}
.hero{padding-bottom:1rem;border-bottom:1px solid #DEE3D7;margin-bottom:1.2rem}
.hero p{color:#526048}.dot{color:#93BA3F}
</style>''',unsafe_allow_html=True)

@st.cache_data
def get_demo(): return demo_data()
@st.cache_resource
def fit_customers(d): return train_customers(d)

def table(d):
    def number(v):
        if pd.isna(v): return '—'
        if isinstance(v, (int, np.integer)): return f'{v:,}'
        if isinstance(v, (float, np.floating)):
            if not np.isfinite(v): return '—'
            if v != 0 and abs(v) < .001: return f'{v:.2e}'
            return f'{v:,.2f}'.rstrip('0').rstrip('.')
        return str(v)
    formats={col:number for col in d.select_dtypes(include='number').columns}
    styled=d.style.format(formats, na_rep='—').set_properties(**{'background-color':'#FFFFFF','color':'#20251B'})
    st.dataframe(styled,width='stretch',hide_index=True)
def chart(fig):
    fig.update_layout(font_color='#20251B',paper_bgcolor='white',plot_bgcolor='white',margin=dict(l=20,r=20,t=40,b=30),legend_title_text='')
    st.plotly_chart(fig,width='stretch',theme=None)
def csv(d):return d.to_csv(index=False).encode('utf-8-sig')

def demo_zip():
    f=io.BytesIO()
    with zipfile.ZipFile(f,'w',zipfile.ZIP_DEFLATED) as z:
        for name,d in get_demo().items():z.writestr(name+'.csv',csv(d))
    return f.getvalue()

st.sidebar.markdown('## MDL<span class="dot">.</span>',unsafe_allow_html=True)
st.sidebar.caption('MARKETING DECISION LAB')
page=st.sidebar.radio('Workspace',['Overview','Customer segments','Targeting & ROI','Machine learning lab','Price & advertising','A/B experiments','Customer lifetime value','Data & methodology'])
st.sidebar.divider()
st.sidebar.caption('A fictional consumer electronics growth case. No API key required.')
reset=st.sidebar.button('Reset demo data',type='primary')
if 'data' not in st.session_state or reset:
    st.session_state.data={k:v.copy() for k,v in get_demo().items()}
    st.session_state.source={k:'Synthetic demo' for k in SCHEMA}
    if reset: st.rerun()
st.sidebar.download_button('Download demo CSV bundle',demo_zip(),'MDL_demo_data.zip','application/zip')
st.sidebar.caption('Upload data in Data & methodology. Uploads are processed on the hosting server in your session; this app makes no external AI/API calls.')

st.markdown('<div class="hero"><div class="eyebrow">FROM MARKETING ANALYTICS TO BUSINESS DECISIONS</div><h1>Marketing Decision Lab<span class="dot">.</span></h1><p>Find the audience. Evaluate the economics. Test the decision.</p></div>',unsafe_allow_html=True)
st.caption('Portfolio demonstration · Synthetic outcomes are not evidence of real commercial performance.')
st.subheader(page)
data=st.session_state.data
try:
    with st.spinner('Training response models…'):
        models,train,test,scores,tree_rules=fit_customers(data['customers'])
except RuntimeError as e:
    st.error(str(e));st.stop()
customers=rfm(data['customers'])

if page in ['Overview','Targeting & ROI','Machine learning lab']:
    st.markdown('**Campaign unit economics**')
    a,b,c=st.columns(3)
    cost=a.number_input('Cost per offer (£)',.01,1000.,2.,step=.1,key='cost')
    goods=b.number_input('Goods revenue per subscriber (£)',0.,5000.,40.,step=1.,key='goods')
    cogs=c.number_input('COGS rate',0.,1.,.60,step=.01,key='cogs')
    a,b=st.columns(2)
    subscription=a.number_input('Subscription fee (£)',0.,1000.,8.99,step=.1,key='subscription')
    shipping=b.number_input('Shipping cost (£)',0.,1000.,5.,step=.5,key='shipping')
    margin=goods*(1-cogs)+subscription-shipping
    st.latex(r'g = Revenue\,(1-COGS) + Subscription\ fee - Shipping\ cost')
    st.caption(f'Contribution per responder = £{margin:.2f}. Default: 40 × (1 − 0.60) + 8.99 − 5 = £19.99.')
    if margin<=0:st.warning('Contribution is nonpositive; no profitable response threshold exists.');st.stop()
    threshold=cost/margin
    policies=[{'Strategy':'Blanket · same test cohort',**campaign(test,np.zeros(len(test)),cost,margin,all_customers=True)}]
    for n,m in models.items():policies.append({'Strategy':n,**campaign(test,m.predict_proba(test[FEATURES])[:,1],cost,margin)})
    policies=pd.DataFrame(policies)

if page=='Overview':
    blanket=campaign(customers,np.zeros(len(customers)),cost,margin,all_customers=True)
    a,b,c,d=st.columns(4)
    a.metric('Synthetic pilot customers',len(customers));b.metric('Blanket campaign ROI',f'{blanket["ROI"]:.1%}')
    c.metric('Training / test',f'{len(train)} / {len(test)}');d.metric('Break-even probability',f'{threshold:.2%}')
    chart(px.bar(policies,x='Strategy',y='Net contribution',color='Strategy',title='Campaign contribution by strategy'))
    st.caption('Blanket ROI uses the entire pilot. Strategy comparisons use the same test cohort. Results describe historical responses and do not measure causal uplift.')
    st.markdown('### Explore your marketing decisions')
    st.write('1. Derive RFM and compare targeted marketing against blanket contact.\n2. Estimate product-week sales using price, marketing expense and brand.\n3. Check experimental group balance and post-treatment activity.\n4. Compute CAC and discounted lifetime value from the loyalty-program funnel.')
    brief='# Marketing Decision Brief\n\nSynthetic data only.\n\n'+policies.to_csv(index=False)+f'\nEntire-pilot blanket ROI: {blanket["ROI"]:.6f}\nBreak-even: {threshold:.6f}\nTraining / test split: 75% / 25%.\nResponse-based ROI is not causal campaign uplift.\n'
    st.download_button('Export decision brief',brief,'Marketing_Decision_Brief.md')

elif page=='Customer segments':
    st.markdown('**Customer segmentation**')
    k=st.slider('Clusters',2,6,3);maximum=st.slider('Maximum iterations',1,30,15)
    features=st.multiselect('Numeric features',FEATURES,default=['recency','monetary_value'])
    if len(features)<2:st.info('Select at least two features.');st.stop()
    @st.cache_data
    def history(d,k,maximum):return cluster_history(d,k,maximum,123)
    h=history(customers[features],k,maximum)
    step=st.slider('Assignment / centroid update step',1,int(h.step.max()),int(h.step.max()))
    view=customers.copy();view['Segment']=h[h.step==step].cluster.to_numpy().astype(str)
    chart(px.scatter(view,x=features[0],y=features[1],color='Segment',opacity=.6,title=f'K-means step {step} · raw selected features'))
    table(view.groupby('Segment')[FEATURES].mean().round(2).reset_index())
    st.caption('Clusters use squared Euclidean distance on raw selected features. Feature scale influences the segmentation. Review the centroid updates to understand each segment.')
    st.download_button('Export segments',csv(view),'segmented_customers.csv')

elif page=='Targeting & ROI':
    st.latex(r'p_{break-even}=\frac{Cost\ per\ offer}{Profit\ per\ subscriber}')
    st.metric('Break-even response probability',f'{threshold:.4%}')
    st.latex(r'ROI=\frac{N_{actual\ responders}\times g-N_{targeted}\times Cost\ per\ offer}{N_{targeted}\times Cost\ per\ offer}')
    table(policies.round(5))
    chart(px.bar(policies,x='Strategy',y='ROI',color='Strategy',title='Response threshold strategy'))
    st.caption('Contact customers whose predicted response probability exceeds break-even. Reported ROI uses observed subscriptions. ROI is undefined when no customers are contacted.')
    with st.expander('Budget allocation scenario'):
        budget=st.number_input('Budget cap (£)',0.,1000000.,700.,step=50.)
        extension=[]
        for n,m in models.items():extension.append({'Strategy':n,**campaign(test,m.predict_proba(test[FEATURES])[:,1],cost,margin,budget=budget)})
        table(pd.DataFrame(extension).round(4))
        st.caption('Eligible customers are ranked by response probability and selected within the budget cap.')
    model_name=st.selectbox('Export policy for',list(models))
    out=test.copy();out['predicted_probability']=models[model_name].predict_proba(test[FEATURES])[:,1];out['target']=out.predicted_probability>threshold
    st.download_button('Export test policy',csv(out),'test_contact_policy.csv')
    st.caption('Historical responses among contacted customers cannot identify incremental campaign uplift without an untreated control.')

elif page=='Machine learning lab':
    a,b,c=st.columns(3);a.metric('Training share','75%');b.metric('Test share','25%');c.metric('Forest trees','5,000')
    table(scores)
    st.caption('Compare decision tree and random forest response predictions on a held-out test cohort.')
    st.markdown('### Simulate a customer with RFM')
    model_name=st.selectbox('Model for simulation',list(models));m=models[model_name]
    a,b,c=st.columns(3)
    recent=a.number_input('Days since last purchase',0,2000,30)
    frequent=b.number_input('Category purchases in past year',0,1000,12)
    money=c.number_input('Electronics + non-electronics spending (£)',0.,100000.,900.,step=50.)
    one=pd.DataFrame([[recent,frequent,money]],columns=FEATURES)
    with st.spinner('Predicting customer response…'):p=m.predict_proba(one)[0,1]
    a,b,c=st.columns(3);a.metric('Predicted response',f'{p:.1%}');b.metric('Expected contribution / contact',f'£{p*margin-cost:.2f}');c.metric('Threshold policy','Target' if p>threshold else 'Do not target')
    st.caption('Single-customer contribution is a model-based estimate. Reported campaign ROI uses actual responders. Inputs are RFM features, not demographic targeting variables.')
    if any(one[f].iloc[0]<train[f].min() or one[f].iloc[0]>train[f].max() for f in FEATURES):st.warning('Profile outside the training range; prediction may be unreliable.')
    grid=pd.DataFrame({'recency':np.linspace(train.recency.min(),train.recency.max(),40),'frequency':frequent,'monetary_value':money})
    grid['Response probability']=m.predict_proba(grid[FEATURES])[:,1]
    fig=px.line(grid,x='recency',y='Response probability',title='Response sensitivity · frequency and spending held constant')
    fig.add_hline(y=threshold,line_dash='dot',line_color='#658B2D');chart(fig)
    with st.expander('Decision tree rules'):st.code(tree_rules)

elif page=='Price & advertising':
    try:d=joined_data(data['products'],data['sales'],data['marketing']);beta,cov,df,levels,coef=fit_mix(d)
    except ValueError as e:st.error(str(e));st.stop()
    st.markdown('**Price and advertising analysis**')
    st.latex(r'final\_price=RRP\,(1-discount)')
    st.latex(r'sales=a+b\,final\_price+c\,marketing\_expense+d\,Brand+\epsilon')
    st.caption('Product-week sales are modeled using price, marketing expense and brand. Philips is the reference brand. Confidence intervals use conventional OLS standard errors.')
    table(coef)
    st.markdown('### Sales scenario simulator')
    product=st.selectbox('Product',d.product_id.unique());row=d[d.product_id==product].iloc[0]
    a,b,c=st.columns(3);disc=a.slider('Discount',0.,.4,.10,.01)
    spend=b.number_input('Brand-week marketing expense (£)',0.,100000.,float(d[d.brand==row.brand].marketing_expense.median()),step=100.)
    xrow=pd.DataFrame([row]);xrow['final_price']=row.RRP*(1-disc);xrow['marketing_expense']=spend
    X=design(xrow,levels)[0];sales=float((X@beta)[0]);se=float(np.sqrt(max(0,(X@cov@X.T)[0,0])));ci=stats.t.ppf(.975,df)*se
    a,b=st.columns(2);a.metric('Predicted product-week units',f'{sales:.1f}');b.metric('95% mean-sales interval',f'{sales-ci:.1f} – {sales+ci:.1f}')
    st.caption('Sales predictions describe model associations. They do not establish the causal effect of a price change.')
    if sales<0 or not d.final_price.min()<=xrow.final_price.iloc[0]<=d.final_price.max() or not d.marketing_expense.min()<=spend<=d.marketing_expense.max():st.warning('Scenario is outside model support or gives negative sales; interpret cautiously.')
    with st.expander('Product attributes and fixed effects'):
        _,_,_,_,q9=fit_mix(d,extended=True);table(q9)
        st.code('sales ~ final_price + factor(screensize) + marketing_expense |\nbrand + technology + resolution + energy_class + support_HDR + refresh_rate')
        st.caption('Categorical effects are estimated with indicator variables. The table displays price, screen-size and advertising coefficients.')
    with st.expander('Joined data and quality checks'):
        table(d.head(30))
        st.write('Products and sales are linked by product ID; marketing records are linked by brand and week. Total marketing spend counts each brand-week record once.')
        st.metric('Total brand-week marketing expense',f'£{data["marketing"].marketing_expense.sum():,.0f}')
    st.download_button('Export joined product-week data',csv(d),'data_TV.csv')

elif page=='A/B experiments':
    st.markdown('**Experiment performance analysis**')
    d=data['experiments'];balance,effects,reg=experiment_results(d)
    st.latex(r'post\_total\_activity=a+b_A\,I(treatment=A)+b_B\,I(treatment=B)+\epsilon')
    summary=d.groupby('treatment',as_index=False).agg(Users=('user_id','size'),Mean_post_activity=('post_total_activity','mean'))
    chart(px.bar(summary,x='treatment',y='Mean_post_activity',color='treatment',title='Post-treatment activity by variant'))
    st.markdown('**1 · Randomization balance checks**');table(balance)
    st.caption('Gender is recoded F=1, M=0. Tests compare age, gender, account age and pre-treatment activity against control. Nonsignificant balance tests do not prove randomization.')
    st.markdown('**2 · Post-activity Welch t-tests**');table(effects)
    st.markdown('**3 · Regression with control reference**');table(reg)
    st.caption('Welch tests allow unequal group variances; regression uses pooled OLS variance, so p-values may differ. P-values are unadjusted for multiple comparisons.')
    st.download_button('Export treatment effects',csv(effects),'activity_treatment_effects.csv')

elif page=='Customer lifetime value':
    st.markdown('**Acquisition economics and customer lifetime value**')
    a,b,c=st.columns(3)
    membership=a.number_input('Annual membership (£)',0.,10000.,89.)
    visits=b.number_input('Visits per year',1,1000,40)
    revenue=c.number_input('Revenue per visit (£)',0.,10000.,100.)
    a,b,c=st.columns(3)
    pm=a.number_input('Product contribution margin rate',0.,1.,.07,step=.01)
    delivery=b.number_input('Delivery cost per visit (£)',0.,10000.,5.)
    promo=c.number_input('First trial order promotion (£)',0.,10000.,10.)
    a,b,c=st.columns(3)
    cpc=a.number_input('Cost per click (£)',0.,1000.,.4,step=.1)
    click_trial=b.number_input('Click → trial probability',.001,1.,.10,step=.01)
    trial_member=c.number_input('Trial → member probability',.001,1.,.20,step=.01)
    a,b,c=st.columns(3)
    retention=a.slider('Annual retention',0.,1.,.70,.01);discount=b.slider('Annual discount rate',0.,.4,.10,.01);N=c.slider('Years',1,15,5)
    v,flows=course_clv(N,membership,visits,revenue,pm,retention,discount,cpc,click_trial,trial_member,promo,delivery)
    st.latex(r'CAC=\frac{CPC}{p_{click\to trial}p_{trial\to member}}+\frac{2\,Delivery}{p_{trial\to member}}-\frac{(Revenue-Promo)-Revenue(1-margin)+Revenue\,margin}{p_{trial\to member}}')
    st.latex(r'M=membership+Revenue\times visits\times margin,\quad c=Delivery\times visits,\quad g=M-c')
    st.latex(r'CLV=\sum_{t=1}^{N}\frac{g\,r^{t-1}}{(1+k)^t}-CAC')
    a,b,c=st.columns(3);a.metric('Computed CAC',f'£{v["CAC"]:.2f}');b.metric('Annual g = M − c',f'£{v["g"]:.2f}');c.metric('Net CLV',f'£{v["CLV"]:.2f}')
    table(pd.DataFrame({'Component':list(v),'Value':list(v.values())}).round(4))
    chart(px.bar(pd.DataFrame({'Year':np.arange(1,N+1),'Discounted expected contribution':flows}),x='Year',y='Discounted expected contribution'))
    st.caption('Acquisition costs include two trial orders and a first-order promotion. Expected annual contribution is adjusted for retention and discounted over the selected horizon.')

else:
    st.markdown('### Upload marketing data')
    st.write('Explore the built-in synthetic datasets or upload CSV files using the required columns. Uploaded files are processed on the hosting server for your session.')
    table(pd.DataFrame([{'Dataset':k,'Rows':len(v),'Source':st.session_state.source[k]} for k,v in data.items()]))
    with st.form('upload'):
        files={k:st.file_uploader(k+' · '+', '.join(SCHEMA[k]),type='csv',key='up_'+k) for k in SCHEMA}
        apply=st.form_submit_button('Validate & apply')
    if apply:
        try:
            pending=data.copy();sources=st.session_state.source.copy()
            for k,f in files.items():
                if f is not None:pending[k]=validate(k,pd.read_csv(f));sources[k]='Uploaded: '+f.name
            joined_data(pending['products'],pending['sales'],pending['marketing'])
            if any(f is not None for f in files.values()):st.session_state.data=pending;st.session_state.source=sources;st.rerun()
            else:st.info('Choose at least one CSV.')
        except (ValueError,pd.errors.ParserError,UnicodeDecodeError) as e:st.error('Not applied: '+str(e))
    st.markdown('### Analysis methodology')
    st.write('Response models use RFM features. Campaign ROI uses observed subscriptions, sales scenarios use OLS regression, experiments compare activity against control, and CLV discounts expected contribution after acquisition costs.')
    st.caption('Built-in data is synthetic and contains no company or personal records. This tool makes no external AI/API calls.')
