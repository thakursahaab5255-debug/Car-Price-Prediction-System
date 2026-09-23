from pathlib import Path
import json, sys, time, warnings
import numpy as np, pandas as pd, joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
warnings.filterwarnings('ignore')

BASE_DIR=Path(__file__).resolve().parent
MODEL_DIR=BASE_DIR/'model'; MODEL_DIR.mkdir(exist_ok=True)
MODEL_FILE=MODEL_DIR/'car_price_model.joblib'; META_FILE=MODEL_DIR/'metadata.json'
TARGET='selling_price_lakh'
FEATURES=['brand','model','variant','body_type','manufacturing_year','vehicle_age','registration_year','kilometers_driven','fuel_type','transmission','engine_cc','max_power_bhp','mileage_kmpl','seats','ownership','seller_type','city','state','color','condition','insurance_type','service_history','accident_history','warranty_available','tyre_condition','features_count']
CAT=['brand','model','variant','body_type','fuel_type','transmission','ownership','seller_type','city','state','color','condition','insurance_type','service_history','accident_history','warranty_available','tyre_condition']
NUM=[x for x in FEATURES if x not in CAT]
REQ=FEATURES+[TARGET]

def find_csv():
    preferred=[BASE_DIR/'car_data.csv',BASE_DIR/'CarPredict_AI_100K_Indian_Used_Cars(1).csv',BASE_DIR/'CarPredict_AI_100K_Indian_Used_Cars.csv',BASE_DIR/'dataset'/'car_data.csv',BASE_DIR/'data'/'car_data.csv']
    for p in preferred:
        if p.exists(): return p
    for p in BASE_DIR.rglob('*.csv'):
        if 'venv' in p.parts or '.venv' in p.parts: continue
        try:
            if TARGET in pd.read_csv(p,nrows=2).columns: return p
        except Exception: pass
    return None

def main():
    print('\n'+'='*68); print('CarPredict AI - MODEL TRAINING'); print('='*68)
    path=find_csv()
    if not path:
        print('ERROR: Dataset not found. Put your CSV in the same folder as train_model.py and name it car_data.csv.'); sys.exit(1)
    print('[1/6] Dataset:',path)
    df=pd.read_csv(path,low_memory=False); print('[2/6] Rows:',len(df),'Columns:',len(df.columns))
    missing=[c for c in REQ if c not in df.columns]
    if missing: print('ERROR: Missing columns:',missing); sys.exit(1)
    print('[3/6] Cleaning...')
    for c in NUM+[TARGET]: df[c]=pd.to_numeric(df[c],errors='coerce')
    for c in CAT: df[c]=df[c].astype('string').str.strip()
    df['accident_history']=df['accident_history'].fillna('No Record').replace('','No Record')
    df['vehicle_age']=2026-df['manufacturing_year']
    df=df.dropna(subset=[TARGET]); df=df[(df[TARGET]>0)&(df[TARGET]<=200)]
    for c in ['kilometers_driven','engine_cc','max_power_bhp','mileage_kmpl','seats','features_count']:
        df.loc[df[c]<0,c]=np.nan
    X=df[FEATURES].copy(); y=df[TARGET].astype(float)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.20,random_state=42)
    print('[4/6] Building pipeline...')
    pre=ColumnTransformer([('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('enc',OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=np.nan,encoded_missing_value=np.nan))]),CAT),('num',SimpleImputer(strategy='median'),NUM)])
    mask=[True]*len(CAT)+[False]*len(NUM)
    reg=HistGradientBoostingRegressor(loss='squared_error',learning_rate=.08,max_iter=250,max_leaf_nodes=31,min_samples_leaf=20,l2_regularization=.5,early_stopping=True,validation_fraction=.12,n_iter_no_change=20,random_state=42,categorical_features=mask)
    pipe=Pipeline([('preprocessor',pre),('model',reg)])
    print('[5/6] Training...'); pipe.fit(Xtr,ytr)
    pred=np.maximum(pipe.predict(Xte),0); r2=float(r2_score(yte,pred)); mae=float(mean_absolute_error(yte,pred)); print('R²:',round(r2,6),'| R² %:',round(r2*100,2),'| MAE:',round(mae,4),'lakh')
    print('[6/6] Saving...'); joblib.dump(pipe,MODEL_FILE,compress=3)
    options={}
    for c in FEATURES:
        if c in CAT: options[c]=sorted(df[c].dropna().astype(str).unique().tolist(),key=str.lower)
        else: options[c]=[]
    brand_to_models={str(b):sorted(g['model'].dropna().astype(str).unique().tolist(),key=str.lower) for b,g in df.groupby('brand',dropna=True)}
    model_to_variants={str(m):sorted(g['variant'].dropna().astype(str).unique().tolist(),key=str.lower) for m,g in df.groupby('model',dropna=True)}
    meta={'model_name':'HistGradientBoostingRegressor','target':TARGET,'features':FEATURES,'r2':round(r2,8),'accuracy_percent':round(r2*100,2),'mae_lakh':round(mae,8),'training_rows':int(len(Xtr)),'test_rows':int(len(Xte)),'dataset_file':path.name,'created_at':time.strftime('%Y-%m-%d %H:%M:%S'),'options':options,'brand_to_models':brand_to_models,'model_to_variants':model_to_variants}
    META_FILE.write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf-8')
    print('\nTRAINING COMPLETE'); print('Model:',MODEL_FILE); print('Website:', 'http://127.0.0.1:5000')
if __name__=='__main__': main()
