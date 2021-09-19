import numpy as np
import pandas as pd
import pickle
import os
import sys
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge,Lasso ,RidgeCV,LassoCV , ElasticNet , ElasticNetCV,LinearRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

'''
reading the dataset
'''
try:
    df = pd.read_csv("ai4i2020.csv",index_col=0)
    
except Exception as e:
        logger.exception(
            "Unable to read CSV, Error: %s", e
        )


df.rename(columns ={'Air temperature [K]':'AT','Process temperature [K]':'PT','Rotational speed [rpm]':'RS',
                        'Torque [Nm]':'Torque','Tool wear [min]':'TW','Machine failure':'MF'}, inplace =True)

df.reset_index(drop = True, inplace = True)

#dropping the columns which are not needed
df.drop(['Product ID', 'Type'], axis =1, inplace = True)

#scaling down to being the data in terms of normal distribution also the algorithm will run faster in this case.
scaler = StandardScaler()
df1 = df.iloc[:,1:] = scaler.fit_transform(df.iloc[:,1:])

'''creating the pandas profiling'''
pf = ProfileReport(df)
#pf.to_widgets()

'''
Observations :-
1) we have 10000 observations wirh 5 numeric and 6 categorical columns
2) NAN values -> there are no nan values.
3) Under overview warnings tab we see that most of the features are highy correlated. however in correlations tab we see that 

'''

'''checking a correlation'''
df.iloc[:,1:].corr()

'''creating heatmap vizualisation'''
sns.heatmap(df.corr())

x = df.iloc[:,1:]
y = df['AT']

from statsmodels.stats.outliers_influence import variance_inflation_factor
vif_df = pd.DataFrame()

vif_df['vif'] = [variance_inflation_factor(df1,i) for i in range(df1.shape[1])]
vif_df['feature']  = x.columns
vif_df

#as machine failure is highly correlated hence dropping the feature.
df.drop(['MF'], inplace =True, axis =1)

x_train, x_test, y_train,y_test = train_test_split(df, y,test_size= 0.20,random_state = 100)

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(x_train, y_train)

lm.coef_
lm.fit(x_test,y_test)
lm.score(x_test,y_test)

from sklearn.metrics import mean_squared_error, r2_score
y_pred= lm.predict(x_test)

print('MSE is', mean_squared_error(y_test, y_pred))

print('R2 score is',r2_score(y_test, y_pred))

import statsmodels.api as sm
x = sm.add_constant(x)
model = sm.OLS(y, x).fit()

print(model.rsquared_adj)

from sklearn.linear_model import Ridge,Lasso ,RidgeCV,LassoCV , ElasticNet , ElasticNetCV,LinearRegression
lassocv = LassoCV(alphas=None,cv= 50 , max_iter=200000, normalize=True)
lassocv.fit(x_train,y_train)


lassocv.alpha_

lasso = Lasso(alpha=lassocv.alpha_)
lasso.fit(x_train,y_train)

lasso.score(x_test,y_test)

ridgecv = RidgeCV(alphas=np.random.uniform(0,10,50),cv = 10 , normalize=True)
ridgecv.fit(x_train,y_train)

ridgecv.alpha_

ridge_lr = Ridge(alpha=ridgecv.alpha_)
ridge_lr.fit(x_train,y_train)

ridge_lr.score(x_test,y_test)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

with mlflow.start_run():
    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    lr.fit(x_train, y_train)

    prediction = lr.predict(x_test)
    (rmse, mae, r2) = eval_metrics(y_test, prediction)
    print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

    mlflow.log_param("alpha", alpha)
    mlflow.log_param("l1_ratio", l1_ratio)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)

    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

    if tracking_url_type_store != "file":
    
        # Register the model
        # There are other ways to use the Model Registry, which depends on the use case,
        # please refer to the doc for more information:
        # https://mlflow.org/docs/latest/model-registry.html#api-workflow
        mlflow.sklearn.log_model(lr, "model", registered_model_name="ElasticnetWineModel")
    else:
        mlflow.sklearn.log_model(lr, "model")


file = 'linear_reg.pkl'
pickle.dump(lm,open(file,'wb'))