#%%
import pandas as pd
from src.utils.validation import initial_train_test_split_temporal
from prophet import Prophet
from tqdm import tqdm
import numpy as np
from src.helper.helper import metric, scale_prediction
from src.helper.helper import add_date_cols
# %%
train_data = pd.read_parquet("data/train_data.parquet")
print(train_data.isna().sum() / len(train_data))

# %%
submission_data = pd.read_parquet("data/submission_data.parquet")
print(submission_data.isna().sum() / len(train_data))

# %%
split_feats = ["brand", "country"]
train_data["code"] = [
    "_".join([str(brand), str(country)])
    for brand, country in zip(train_data["brand"], train_data["country"])
]
submission_data["code"] = [
    "_".join([str(brand), str(country)])
    for brand, country in zip(submission_data["brand"], submission_data["country"])
]
#%%
unique_codes_test = submission_data.code.unique().tolist()
train_data = train_data.query('code in @unique_codes_test and date>"2017-01-01"')

# %%
keep_cols =['code','date','wd','dayweek','n_nwd_bef','n_nwd_aft','wd_perc']
train_data = train_data[keep_cols+['phase']].rename(columns = {'date':'ds',
                                          'phase':'y'})
submission_data = submission_data[keep_cols].rename(columns = {'date':'ds'})

#%%
predictions_tr = {'code':[],
               'prediction':[],
               'prediction_lower':[],
               'prediction_upper':[],
               'date':[]}
predictions_te = {'code':[],
               'prediction':[],
               'prediction_lower':[],
               'prediction_upper':[],
               'date':[]}
predictions = {'code':[],
               'prediction':[],
               'prediction_lower':[],
               'prediction_upper':[],
               'date':[]}

def format_df_for_prophet(df):
    df = df.set_index('ds')
    idx = pd.date_range(df.index.min(), df.index.max())
    df = df.reindex(idx)
    df['exo_non_ex_ampl'] = df['wd'].isna().rolling(window = 7).median()
    df['exo_non_ex'] = df['wd'].isna()
    df.fillna(df.median(),inplace = True)
    df = df.reset_index().rename(columns = {'index':'ds'})
    return df
#%%%

# {'changepoint_prior_scale': 0.05,
#  'seasonality_prior_scale': 0.5,
#  'seasonality_mode': 'multiplicative',
#  'changepoint_range': 0.95,
#  'n_changepoints': 25}
for code,tmp_df in tqdm(train_data.groupby('code',as_index = False)):
    tmp_df = format_df_for_prophet(tmp_df.drop(columns ='code'))
    X = tmp_df
    y = tmp_df
    X_tr, X_te, _, _ = initial_train_test_split_temporal(X, y,date_col = 'ds')
    model = Prophet(daily_seasonality=False,
                    weekly_seasonality=True,
                    yearly_seasonality=True) 
    model.fit(X_tr)
    predictions_train = model.predict(X_tr)
    predictions_test = model.predict(X_te)
    
    model = Prophet(daily_seasonality=False,
                    weekly_seasonality=True,
                    yearly_seasonality=True)
    model.fit(X)
    #Regenerar para pillar las exogenas
    predictions_train = model.predict(X_tr)
    predictions_test = model.predict(X_te)
    tmp_sumb = submission_data.query(f'code=="{code}"')
    tmp_sumb = format_df_for_prophet(tmp_sumb.drop(columns ='code'))
    prediction_subm = model.predict(tmp_sumb)
    
    predictions_tr['code']+=[code]*len(X_tr)
    predictions_te['code']+=[code]*len(X_te)
    predictions['code']+=[code]*len(tmp_sumb)
    
    
    predictions_tr['prediction']+=np.clip(0,predictions_train.yhat,np.inf).tolist()
    predictions_te['prediction']+=np.clip(0,predictions_test.yhat,np.inf).tolist()
    predictions['prediction']+=np.clip(0,prediction_subm.yhat,np.inf).tolist()
    
    predictions_tr['prediction_lower']+=np.clip(0,predictions_train.yhat_lower,np.inf).tolist()
    predictions_te['prediction_lower']+=np.clip(0,predictions_test.yhat_lower,np.inf).tolist()
    predictions['prediction_lower']+=np.clip(0,prediction_subm.yhat_lower,np.inf).tolist()
    
    
    predictions_tr['prediction_upper']+=np.clip(0,predictions_train.yhat_upper,np.inf).tolist()
    predictions_te['prediction_upper']+=np.clip(0,predictions_test.yhat_upper,np.inf).tolist()
    predictions['prediction_upper']+=np.clip(0,prediction_subm.yhat_upper,np.inf).tolist()
    
    predictions_tr['date']+=predictions_train.ds.tolist()
    predictions_te['date']+=predictions_test.ds.tolist()
    predictions['date']+=prediction_subm.ds.tolist()

# %%
predictions_tr =pd.DataFrame(predictions_tr)
predictions_te = pd.DataFrame(predictions_te)
predictions = pd.DataFrame(predictions)
#%%
predictions.to_csv('predictions_prophet_exogenous.csv')
predictions_tr.to_csv('predictions_prophet_tr_exogenous.csv')
predictions_te.to_csv('predictions_prophet_te_exogenous.csv')

#%%
print ('done')
import sys
sys.exit()
#%%
predictions  =pd.read_csv('predictions_prophet.csv')
predictions_tr=pd.read_csv('predictions_prophet_tr.csv')
predictions_te=pd.read_csv('predictions_prophet_te.csv')
#%%
train_data = pd.read_parquet("data/train_data.parquet")
print(train_data.isna().sum() / len(train_data))

# %%
submission_data = pd.read_parquet("data/submission_data.parquet")
print(submission_data.isna().sum() / len(train_data))

# %%
split_feats = ["brand", "country"]
train_data["code"] = [
    "_".join([str(brand), str(country)])
    for brand, country in zip(train_data["brand"], train_data["country"])
]


print(f'Unique brands: {train_data["brand"].nunique()}')
print(f'Unique countries: {train_data["country"].nunique()}')
print(f'Unique tuples: {train_data["code"].nunique()}')

# %%

submission_data["code"] = [
    "_".join([str(brand), str(country)])
    for brand, country in zip(submission_data["brand"], submission_data["country"])
]

unique_codes_test = submission_data.code.unique().tolist()
print(f'Unique brands in submission: {submission_data["brand"].nunique()}')
print(f'Unique countries in submission: {submission_data["country"].nunique()}')
print(f'Unique tuples in submission: {submission_data["code"].nunique()}')
# %%
train_data = train_data.query('code in @unique_codes_test and date>"2017-01-01"')

y = train_data["phase"]
X = train_data.copy()



X_tr, X_te, y_tr, y_te = initial_train_test_split_temporal(X, y)
#%%
predictions_tr['date'] = pd.to_datetime(predictions_tr['date'])
predictions_te['date'] = pd.to_datetime(predictions_te['date'])
        
#%%
X_tr = X_tr.merge(predictions_tr,on = ['code','date'],how = 'inner')
X_te = X_te.merge(predictions_te,on = ['code','date'],how = 'inner')
#%%

X_tr = add_date_cols(X_tr,add_weights = False)
X_te = add_date_cols(X_te,add_weights = False)
submission_data = add_date_cols(submission_data,add_weights = False)
#%%
X_tr = scale_prediction(X_tr)
X_te = scale_prediction(X_te)
#%%
X_te.loc[X_te["sum_pred"]==0,'prediction'] = 1
X_te = scale_prediction(X_te)
#%%
X_tr.loc[X_tr["sum_pred"]==0,'prediction'] = 1
X_tr = scale_prediction(X_tr)
#%%
metric(X_tr)
# %%
metric(X_te)
#%%
submission_data = submission_data.merge(predictions,on = ['code','date'],how = 'inner')
#%%
submission_data = scale_prediction(submission_data)
#%%
print(submission_data.loc[submission_data["sum_pred"]==0,'prediction'])
#%%
submission_data.loc[submission_data["sum_pred"]==0,'prediction'] = 1
submission_data = scale_prediction(submission_data)
#%%
submission_template = pd.read_csv("data/submission_template.csv")
submission_data = submission_data[submission_template.keys()]
# %%
# Save Submission
sub_number = "_lags_fbp"
sub_name = "submission/submission{}.csv".format(sub_number)
submission_data.to_csv(sub_name, index=False)
