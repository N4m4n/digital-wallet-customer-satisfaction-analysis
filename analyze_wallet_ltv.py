#!/usr/bin/env python3
"""Analyze the digital-wallet customer lifetime value dataset.
Run: python analyze_wallet_ltv.py --input digital_wallet_ltv_dataset.csv --output output
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',default='digital_wallet_ltv_dataset.csv')
    parser.add_argument('--output',default='output')
    args=parser.parse_args()
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(args.input)
    original_rows=len(df)
    duplicate_rows=int(df.duplicated().sum())
    df=df.drop_duplicates().copy()
    target='Customer_Satisfaction_Score'
    continuous=['Total_Transactions','Active_Days','Loyalty_Points_Earned','Cashback_Received','Support_Tickets_Raised','Issue_Resolution_Time']
    categorical=['App_Usage_Frequency','Preferred_Payment_Method']
    predictors=continuous+categorical
    model_df=df[[target]+predictors].dropna().copy()
    X=pd.get_dummies(model_df[predictors],columns=categorical,drop_first=True,dtype=float)
    for col in continuous:
        sd=X[col].std()
        if sd != 0: X[col]=(X[col]-X[col].mean())/sd
    X=sm.add_constant(X)
    y=model_df[target]
    model=sm.OLS(y,X).fit()
    coeff=pd.DataFrame({'coefficient':model.params,'std_error':model.bse,'t_value':model.tvalues,'p_value':model.pvalues,'ci_low':model.conf_int()[0],'ci_high':model.conf_int()[1]})
    coeff.to_csv(out/'regression_coefficients.csv')
    df.groupby('App_Usage_Frequency')[target].agg(['count','mean','std']).to_csv(out/'satisfaction_by_usage.csv')
    df.groupby('Preferred_Payment_Method')[target].agg(['count','mean','std']).sort_values('mean',ascending=False).to_csv(out/'satisfaction_by_payment.csv')
    df[continuous+[target]].corr()[target].drop(target).sort_values().to_csv(out/'correlations_with_satisfaction.csv',header=['correlation'])
    plt.figure(figsize=(8,5)); sns.histplot(df[target],bins=10,discrete=True); plt.title('Distribution of Customer Satisfaction'); plt.xlabel('Satisfaction score'); plt.tight_layout(); plt.savefig(out/'satisfaction_distribution.png',dpi=180); plt.close()
    order=['Monthly','Weekly','Daily']
    plt.figure(figsize=(8,5)); sns.barplot(data=df,x='App_Usage_Frequency',y=target,order=[x for x in order if x in df.App_Usage_Frequency.unique()],errorbar=None); plt.title('Average Satisfaction by App Usage Frequency'); plt.xlabel('App usage frequency'); plt.ylabel('Mean satisfaction'); plt.tight_layout(); plt.savefig(out/'satisfaction_by_usage.png',dpi=180); plt.close()
    plt.figure(figsize=(8,5)); sns.scatterplot(data=df,x='Issue_Resolution_Time',y=target,alpha=.25,s=15); sns.regplot(data=df,x='Issue_Resolution_Time',y=target,scatter=False,color='red'); plt.title('Issue Resolution Time and Satisfaction'); plt.tight_layout(); plt.savefig(out/'resolution_vs_satisfaction.png',dpi=180); plt.close()
    plt.figure(figsize=(9,7)); sns.heatmap(df[continuous+[target]].corr(),annot=True,fmt='.2f',cmap='coolwarm',center=0); plt.title('Correlation Heatmap'); plt.tight_layout(); plt.savefig(out/'correlation_heatmap.png',dpi=180); plt.close()
    with open(out/'analysis_summary.txt','w',encoding='utf-8') as f:
        f.write(f'Original rows: {original_rows}\nRows after duplicate removal: {len(df)}\nDuplicate rows removed: {duplicate_rows}\nColumns: {df.shape[1]}\nMissing cells: {int(df.isna().sum().sum())}\n')
        f.write(f'Mean satisfaction: {df[target].mean():.4f}\nMedian satisfaction: {df[target].median():.4f}\nStd satisfaction: {df[target].std():.4f}\n')
        f.write(f'R-squared: {model.rsquared:.6f}\nAdjusted R-squared: {model.rsquared_adj:.6f}\nF-test p-value: {model.f_pvalue:.6g}\n\n{model.summary()}\n')
    print(model.summary())
    print(f'Analysis outputs saved in {out.resolve()}')
if __name__=='__main__': main()
