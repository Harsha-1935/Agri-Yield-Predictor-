import os, zipfile, warnings
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE=os.path.dirname(os.path.abspath(__file__)); OUT=os.path.join(BASE,'Agri_Yield_Outputs'); os.makedirs(OUT,exist_ok=True)
BUNDLE=os.path.join(BASE,'Agri_Yield_Project_Data_Bundle.zip')
DATAFILE='Agri_Yield_Project_Data_Bundle/original_dataset/crop_yield_dataset.csv'

def load_data():
    if not os.path.exists(BUNDLE): raise FileNotFoundError('Keep Agri_Yield_Project_Data_Bundle.zip beside this script.')
    with zipfile.ZipFile(BUNDLE) as z: df=pd.read_csv(z.open(DATAFILE))
    df.columns=[str(c).strip() for c in df.columns]
    mapping={'Rainfall':'Rainfall','Nitrogen':'Nitrogen','Phosphorus':'Phosphorus','Potassium':'Potassium','State':'Region','Crop':'Crop','Yield':'Yield'}
    df=df.rename(columns={v:k for k,v in mapping.items() if v in df.columns})
    needed=['Rainfall','Nitrogen','Phosphorus','Potassium','State','Crop','Yield']
    missing=[c for c in needed if c not in df.columns]
    if missing: raise ValueError('Dataset columns missing: '+str(missing))
    df=df[needed].replace([np.inf,-np.inf],np.nan).dropna(subset=['Yield'])
    for c in ['Rainfall','Nitrogen','Phosphorus','Potassium','Yield']: df[c]=pd.to_numeric(df[c],errors='coerce')
    df=df.dropna(subset=['Rainfall','Nitrogen','Phosphorus','Potassium','Yield'])
    return df

def pipeline():
    num=['Rainfall','Nitrogen','Phosphorus','Potassium']; cat=['State','Crop']
    pre=ColumnTransformer([('n',SimpleImputer(strategy='median'),num),('c',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('oh',OneHotEncoder(handle_unknown='ignore'))]),cat)])
    return pre

def plots(df, metrics, best, Xtest,ytest,pred):
    sns.set_theme(style='whitegrid')
    # Two plots per window (side-by-side)
    fig, ax = plt.subplots(1,2,figsize=(14,5))
    sns.histplot(df.Yield,bins=40,ax=ax[0]); ax[0].set_title('Yield Distribution')
    sns.scatterplot(data=df,x='Rainfall',y='Yield',alpha=.35,ax=ax[1]); ax[1].set_title('Rainfall vs Yield')
    fig.tight_layout(); fig.savefig(os.path.join(OUT,'01_yield_distribution_and_rainfall.png')); plt.show()

    fig, ax = plt.subplots(1,2,figsize=(14,6))
    sns.heatmap(df[['Rainfall','Nitrogen','Phosphorus','Potassium','Yield']].corr(),annot=True,cmap='coolwarm',ax=ax[0]); ax[0].set_title('Correlation Heatmap')
    df.groupby('Crop').Yield.mean().sort_values().tail(15).plot(kind='barh',ax=ax[1]); ax[1].set_title('Average Yield by Crop (Top 15)')
    fig.tight_layout(); fig.savefig(os.path.join(OUT,'02_correlation_and_crop.png')); plt.show()

    fig, ax = plt.subplots(1,2,figsize=(14,6))
    df.groupby('State').Yield.mean().sort_values().tail(15).plot(kind='barh',ax=ax[0]); ax[0].set_title('Average Yield by State (Top 15)')
    ax[1].scatter(ytest,pred,alpha=.4); lo=min(ytest.min(),pred.min()); hi=max(ytest.max(),pred.max()); ax[1].plot([lo,hi],[lo,hi]); ax[1].set_xlabel('Actual'); ax[1].set_ylabel('Predicted'); ax[1].set_title('Actual vs Predicted')
    fig.tight_layout(); fig.savefig(os.path.join(OUT,'03_state_and_actual_predicted.png')); plt.show()

    fig, ax = plt.subplots(1,2,figsize=(14,5))
    ax[0].scatter(pred,ytest-pred,alpha=.4); ax[0].axhline(0); ax[0].set_title('Residual Plot'); ax[0].set_xlabel('Predicted'); ax[0].set_ylabel('Residual')
    metrics.set_index('Model')[['MAE','RMSE']].plot(kind='bar',ax=ax[1]); ax[1].set_title('Model Comparison (MAE/RMSE)')
    fig.tight_layout(); fig.savefig(os.path.join(OUT,'04_residuals_and_model_comparison.png')); plt.show()

def main():
    print('Loading dataset...'); df=load_data(); print('Dataset loaded:',len(df),'rows')
    if len(df)>15000: df=df.sample(15000,random_state=42)
    X=df[['Rainfall','Nitrogen','Phosphorus','Potassium','State','Crop']]; y=df.Yield
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
    models={'Linear Regression':LinearRegression(),'Decision Tree':DecisionTreeRegressor(max_depth=12,min_samples_leaf=3,random_state=42),'Random Forest':RandomForestRegressor(n_estimators=60,min_samples_leaf=2,random_state=42,n_jobs=-1)}
    results=[]; fitted={}
    for name,est in models.items():
        model=Pipeline([('prep',pipeline()),('model',est)]); print('Training',name); model.fit(Xtr,ytr); p=model.predict(Xte)
        results.append({'Model':name,'MAE':mean_absolute_error(yte,p),'RMSE':np.sqrt(mean_squared_error(yte,p)),'R2':r2_score(yte,p)});fitted[name]=model
    m=pd.DataFrame(results).sort_values('RMSE'); m.to_csv(os.path.join(OUT,'model_metrics.csv'),index=False); print(m.to_string(index=False))
    bestname=m.iloc[0].Model; best=fitted[bestname]; pred=best.predict(Xte); plots(df,m,best,Xte,yte,pred)
    best.fit(X,y)
    import joblib; joblib.dump(best,os.path.join(OUT,'yield_model.joblib'))
    print('\nBest model:',bestname,'\nPlots and metrics saved in',OUT)
    farmer_interface(best,df)

def farmer_interface(model,df):
    print("\n--- FARMER INPUT: CROP YIELD PREDICTION ---")
    print("Enter the six values below. State and Crop can be typed normally.\n")
    row={}
    for c in ['Rainfall','Nitrogen','Phosphorus','Potassium']:
        while True:
            try:
                value=float(input(f"{c}: "))
                if value<0: print('Please enter a non-negative value.'); continue
                row[c]=value; break
            except ValueError: print('Please enter a valid number.')
    for c in ['State','Crop']:
        options=sorted(df[c].dropna().astype(str).unique())
        print(f"\nExamples of available {c} values: " + ", ".join(options[:20]))
        value=input(f"{c} (type normally): ").strip()
        match=next((v for v in options if v.strip().casefold()==value.casefold()),None)
        row[c]=match if match is not None else value
    value=model.predict(pd.DataFrame([row]))[0]
    print('\n=====================================')
    print(f" PREDICTED CROP YIELD: {value:.2f} (dataset yield units)")
    print('=====================================\n')
    again=input('Predict for another crop/farm? (y/n): ').strip().lower()
    if again=='y': farmer_interface(model,df)

if __name__=='__main__':
    try:main()
    except Exception as e: print('\nERROR:',e); input('Press Enter to close...')
