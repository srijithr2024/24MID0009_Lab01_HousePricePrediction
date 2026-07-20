#!/usr/bin/env python
# coding: utf-8

# # House Price Prediction using Regression Models
# 
# **Name :** SRIJITH R
# 
# **Reg. no. :** 24MID0009
# 
# **Course code :** MDI3003
# 
# **Course Title :** Advanced Predictive Analytics 
# 
# **Faculty :** Dr. Durgesh Kumar
# 
# **Datasets Used :**
# 1. California Housing Dataset
# 2. Ames Housing Dataset

# **Problem statement :**
# House price predition is the important regression problem in ML.The objective of the LAB 01(House price prediction using Linear regression) is to predict house prices using the property features from the california dataset and ames dataset.Simple linear regression is to used to predict house prices using single feature whereas multiple linear regression is used to predict house prices using one or more features.The preformance of the both models are measured using performance metrices such as MSE,MAE,RMSE,R2 score.

# **Objectives :** 1.) The objective is to learn a function that maps property attributes to a continuous price or price-per-unit-area target.
# 2.)The objective is to build a machine learning model that accurately predicts house prices based on these features. 
# 

# **Success Criteria :**
# 
# 
# 1. The model should achieve lower MAE and RMSE than the baseline model.
# 2. The model should generalize well without overfitting.
# 3. The residuals should not show any systematic pattern.
# 4. The prediction results should be meaningful for real-world use.
# 5.  The trained model should be reproducible and reusable.

# **Step 1-Environment & impot libraries :**
# 

# In[1]:


import platform 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import sklearn
SEED = 42
np.random.seed(SEED)
print("Python:",platform.python_version())
print("pandas:",pd.__version__)
print("Scikit-learn:", sklearn.__version__)


# **Step 2-Loading California Dataset**

# In[2]:


# Option A: California Housing
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing(as_frame=True)
c_df = housing.frame.rename(columns={'MedHouseVal': 'Price'}) 
print(housing.DESCR[:1000])
print(c_df.head())


# **Step 2-Loading Ames dataset**

# In[3]:


# Option C: Ames / Kaggle House Prices
# Download the CSV using the official course instructions and keep raw data unchanged. 
a_df = pd.read_csv(r'C:\Users\sriji\Downloads\ameshousing.csv')
print(a_df.head()) 
print(a_df.shape)


# **Step 3-Data audit for Ames hosuing dataset**

# In[4]:


target_a='SalePrice' # replace for the selected dataset
print('Shape:', a_df.shape)
print('Target dtype:',a_df[target_a].dtype) 
print(a_df.head())
print(a_df.sample(min(5, len(a_df)),random_state=SEED))

audit = pd.DataFrame({ 'dtype': a_df.dtypes.astype(str),'missing_n': a_df.isna().sum(),
'missing_pct': 100 * a_df.isna().mean(),'unique_n': a_df.nunique(dropna=False)
}).sort_values('missing_pct',ascending=False)
print(audit.head(20))
print('Duplicate rows:',a_df.duplicated().sum()) 
print(a_df[target_a].describe())


# **Step 3-Data audit California House pricing dataset**

# In[5]:


target_c= 'Price' # replace for the selected dataset

print('Shape:', c_df.shape)
print('Target dtype:',c_df[target_c].dtype) 
print(c_df.head())
print(c_df.sample(min(5, len(c_df)),random_state=SEED))

audit = pd.DataFrame({ 'dtype': c_df.dtypes.astype(str),'missing_n': c_df.isna().sum(),
'missing_pct': 100 * c_df.isna().mean(),'unique_n': c_df.nunique(dropna=False)
}).sort_values('missing_pct',ascending=False)
print(audit.head(20))
print('Duplicate rows:',c_df.duplicated().sum()) 
print(c_df[target_c].describe())


# **Step 4-Calfornia housing house hold split**

# In[6]:


from sklearn.model_selection import train_test_split
DROP_COLS = ['Id'] if 'Id' in c_df.columns else [] 
X = c_df.drop(columns=[target_c] + DROP_COLS) 
y = c_df[target_c].copy()
X_train_c,X_test_c,y_train_c,y_test_c=train_test_split(X,y,test_size=0.20,random_state=SEED)
print(X_train_c.shape,X_test_c.shape) 
print(y_train_c.shape,y_test_c.shape)


# **Step 4-Ames hosuing dataset house hold split**

# In[7]:


from sklearn.model_selection import train_test_split
target='SalePrice'
DROP_COLS = ['Id'] if 'Id' in a_df.columns else [] 
X = a_df.drop(columns=[target_a] + DROP_COLS) 
y = a_df[target_a].copy()
X_train_a,X_test_a,y_train_a,y_test_a=train_test_split(X,y,test_size=0.20,random_state=SEED)
print(X_train_a.shape,X_test_a.shape) 
print(y_train_a.shape,y_test_a.shape)


# **Step 5-EDA(california dataset)**

# In[8]:


train_df = X_train_c.copy() 
train_df['Price'] = y_train_c
# Target distribution
fig, ax = plt.subplots(figsize=(7, 4)) 
sns.histplot(train_df['Price'], kde=True, ax=ax) 
ax.set_title('California hosuing price Distribution')
plt.show()
# Numerical correlations
num_df = train_df.select_dtypes(include=np.number) 
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(num_df.corr(), cmap='coolwarm', center=0, ax=ax) 
ax.set_title('Numerical Correlation Matrix')
plt.show()


# **Step 5-EDA(Ames dataset)**

# In[9]:


train_df = X_train_a.copy() 
train_df['SalePrice'] = y_train_a
# Target distribution
fig, ax = plt.subplots(figsize=(7, 4)) 
sns.histplot(train_df['SalePrice'], kde=True, ax=ax) 
ax.set_title('Ames hosuing price Distribution')
plt.show()
# Numerical correlations
num_df = train_df.select_dtypes(include=np.number) 
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(num_df.corr(), cmap='coolwarm', center=0, ax=ax) 
ax.set_title('Numerical Correlation Matrix')
plt.show()


# **Step 6-Build a leakage-safe preprocessing pipeline(california dataset)**

# In[10]:


from sklearn.compose import ColumnTransformer 
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
numeric_features_c = X_train_c.select_dtypes(include=np.number).columns.tolist()
categorical_features_c = X_train_c.select_dtypes(exclude=np.number).columns.tolist()
numeric_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')),('scaler', StandardScaler())])
categorical_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
preprocess_c = ColumnTransformer([('num', numeric_pipe, numeric_features_c),('cat', categorical_pipe, categorical_features_c)])


# **Step 6-Build a leakage-safe preprocessing pipeline(ames dataset)**

# In[11]:


from sklearn.compose import ColumnTransformer 
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
numeric_features_a = X_train_a.select_dtypes(include=np.number).columns.tolist()
categorical_features_a = X_train_a.select_dtypes(exclude=np.number).columns.tolist()
numeric_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')),('scaler', StandardScaler())])
categorical_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
preprocess_a = ColumnTransformer([('num', numeric_pipe, numeric_features_a),('cat', categorical_pipe, categorical_features_a)])


# **Step 7-Establish a naive baseline(California Datset)**

# In[12]:


from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
naive = DummyRegressor(strategy='mean') 
naive.fit(X_train_c[numeric_features_c], y_train_c) 
naive_pred = naive.predict(X_test_c[numeric_features_c])
print('Naive MAE:',mean_absolute_error(y_test_c, naive_pred)) 
print('Naive RMSE:',np.sqrt(mean_squared_error(y_test_c, naive_pred)))


# **Step 7-Establish a naive baseline(Ames Dataset)**

# In[13]:


from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
naive = DummyRegressor(strategy='mean') 
naive.fit(X_train_a[numeric_features_a],y_train_a) 
naive_pred = naive.predict(X_test_a[numeric_features_a])
print('Naive MAE:',mean_absolute_error(y_test_a,naive_pred)) 
print('Naive RMSE:',np.sqrt(mean_squared_error(y_test_a, naive_pred)))


# **Step 8-Simple Linear regression(CALIFORNIA DATASET)**

# In[14]:


from sklearn.linear_model import LinearRegression
simple_feature_c = 'MedInc' # replace for the dataset 
simple_model_c = LinearRegression() 
simple_model_c.fit(X_train_c[[simple_feature_c]], y_train_c)
simple_pred_c = simple_model_c.predict(X_test_c[[simple_feature_c]])
print('Intercept:', simple_model_c.intercept_) 
print('Slope:', simple_model_c.coef_[0])


# **Step 8-Simple Linear regression(AMES DATASET)**

# In[15]:


from sklearn.linear_model import LinearRegression
simple_feature_a = 'Gr Liv Area' # replace for the dataset 
simple_model_a = LinearRegression() 
simple_model_a.fit(X_train_a[[simple_feature_a]], y_train_a)
simple_pred_a = simple_model_a.predict(X_test_a[[simple_feature_a]])
print('Intercept:', simple_model_a.intercept_) 
print('Slope:', simple_model_a.coef_[0])


# **Step 9-Mulitple linear regression(california dataset)**

# In[16]:


linear_pipeline_c = Pipeline([("preprocess",preprocess_c),("model",LinearRegression())])
linear_pipeline_c.fit(X_train_c, y_train_c)
linear_pred_c = linear_pipeline_c.predict(X_test_c)


# **Step 9-Mulitple linear regression(ames dataset)**

# In[17]:


linear_pipeline_a = Pipeline([("preprocess",preprocess_a),("model",LinearRegression())])
linear_pipeline_a.fit(X_train_a, y_train_a)
linear_pred_a = linear_pipeline_a.predict(X_test_a)


# **Step 10-reusable evaluation function**
# 

# In[18]:


def evaluate_regressor(name,fitted_model,X_eval,y_eval):
    pred = fitted_model.predict(X_eval)
    mae = mean_absolute_error(y_eval, pred)
    mse = mean_squared_error(y_eval, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_eval, pred)
    Xt = fitted_model.named_steps["preprocess"].transform(X_eval)
    n,p = Xt.shape
    adj_r2 = np.nan if n <= p + 1 else 1 - (1-r2)*(n-1)/(n-p-1)
    return {
        'Model': name,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'Adjusted R2': adj_r2
    }, pred


# **Step 10-COmparing regression models such as(Linear regression,Ridge,Elastic net,Random forest,Gradient Boosting) for the California Housing Dataset**

# In[19]:


from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
model_specs_c = {
    'Linear Regression':LinearRegression(),
    'Ridge':Ridge(alpha=1.0),
    'Elastic Net':ElasticNet(alpha=0.001,l1_ratio=0.5,max_iter=20000),
    'Random Forest':RandomForestRegressor(n_estimators=300,min_samples_leaf=2,random_state=SEED,n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(random_state=SEED)
}
fitted_models_c = {}
results_c = []
predictions_c = {}
for name, estimator in model_specs_c.items():
    pipe = Pipeline([('preprocess', preprocess_c),('model',estimator)])
    pipe.fit(X_train_c,y_train_c)
    row,pred = evaluate_regressor(name, pipe,X_test_c,y_test_c)
    results_c.append(row)
    fitted_models_c[name] =pipe
    predictions_c[name] =pred
results_df_c = pd.DataFrame(results_c).sort_values('RMSE')
print(results_df_c)


# **Step 10-COmparing regression models such as(Linear regression,Ridge,Elastic net,Random forest,Gradient Boosting) for the Ames Housing Dataset**

# In[20]:


model_specs_a = {
    'Linear Regression':LinearRegression(),
    'Ridge':Ridge(alpha=1.0),
    'Elastic Net':ElasticNet(alpha=0.001,l1_ratio=0.5,max_iter=20000),
    'Random Forest':RandomForestRegressor(n_estimators=300,min_samples_leaf=2,random_state=SEED,n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(random_state=SEED)
}
fitted_models_a = {}
results_a = []
predictions_a = {}
for name,estimator in model_specs_a.items():
    pipe = Pipeline([('preprocess', preprocess_a),('model',estimator)])
    pipe.fit(X_train_a,y_train_a)
    row,pred = evaluate_regressor(name, pipe,X_test_a,y_test_a)
    results_a.append(row)
    fitted_models_a[name] =pipe
    predictions_a[name] =pred
results_df_a = pd.DataFrame(results_a).sort_values('RMSE')
print(results_df_a)


# **Step 12-Polynomial Regression(California dataset)**

# In[21]:


from sklearn.preprocessing import PolynomialFeatures
selected_numeric_c = ['MedInc','HouseAge','AveRooms']
poly_pipe_c = Pipeline([
    ('imputer',SimpleImputer(strategy='median')),
    ('poly',PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler',StandardScaler()),
    ('model',Ridge(alpha=1.0))
])
poly_pipe_c.fit(X_train_c[selected_numeric_c],y_train_c)
poly_pred_c = poly_pipe_c.predict(X_test_c[selected_numeric_c])


# **Step 12-Polynomial regression(Ames dataset)**

# In[22]:


selected_numeric_a = ['Gr Liv Area','Overall Qual','Year Built']
poly_pipe_a = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler()),
    ('model', Ridge(alpha=1.0))
])
poly_pipe_a.fit(X_train_a[selected_numeric_a],y_train_a)
poly_pred_a = poly_pipe_a.predict(X_test_a[selected_numeric_a])


# **Step 13-Cross-validation(California dataset)**

# In[23]:


from sklearn.model_selection import KFold, cross_validate
cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
scoring = {'mae':'neg_mean_absolute_error','rmse':'neg_root_mean_squared_error','r2':'r2'}
cv_rows = []
for name,estimator in model_specs_c.items():
    pipe = Pipeline([('preprocess', preprocess_c),('model', estimator)])
    scores = cross_validate(pipe,X_train_c,y_train_c,cv=cv,scoring=scoring,n_jobs=-1)
    cv_rows.append({'Model': name,'Key parameters':estimator.get_params(),'CV MAE mean':-scores['test_mae'].mean(),'CV RMSE mean':-scores['test_rmse'].mean(),'CV RMSE SD': scores['test_rmse'].std(),'CV R² mean':scores['test_r2'].mean(),'Fit time':scores['fit_time'].mean()})
cv_results_c = pd.DataFrame(cv_rows).sort_values('CV RMSE mean')
print(cv_results_c)


# **Step 13-Cross-validation(Ames dataset)**

# In[24]:


cv = KFold(n_splits=5,shuffle=True,random_state=SEED)
scoring = {'mae':'neg_mean_absolute_error','rmse':'neg_root_mean_squared_error','r2':'r2'}
cv_rows = []
for name, estimator in model_specs_a.items():
    pipe = Pipeline([('preprocess', preprocess_a),('model',estimator)])
    scores = cross_validate(pipe,X_train_a,y_train_a,cv=cv,scoring=scoring,n_jobs=-1)
    cv_rows.append({'Model': name,'Key parameters':estimator.get_params(),'CV MAE mean':-scores['test_mae'].mean(),'CV RMSE mean':-scores['test_rmse'].mean(),'CV RMSE SD': scores['test_rmse'].std(),'CV R² mean':scores['test_r2'].mean(),'Fit time':scores['fit_time'].mean()})
cv_results_a = pd.DataFrame(cv_rows).sort_values('CV RMSE mean')
print(cv_results_a)


# **Step 14-Ridge hyperparameter tuning(california dataset)**

# In[25]:


from sklearn.model_selection import GridSearchCV
ridge_pipe_c = Pipeline([
    ('preprocess',preprocess_c),
    ('model',Ridge())
])
ridge_grid = {
    'model__alpha':np.logspace(-3, 3, 13)
}
search_c = GridSearchCV(
    ridge_pipe_c,
    param_grid=ridge_grid,
    scoring='neg_root_mean_squared_error',
    cv=cv,
    n_jobs=-1,
    return_train_score=True
)
search_c.fit(X_train_c,y_train_c)
print('Best Parameters:',search_c.best_params_)
print('Best CV RMSE:',-search_c.best_score_)
best_ridge_c = search_c.best_estimator_


# **Step 14-Ridge hyperparameter tuning(Ames dataset)**

# In[26]:


from sklearn.model_selection import GridSearchCV
ridge_pipe_a = Pipeline([
    ('preprocess',preprocess_a),
    ('model',Ridge())
])
ridge_grid = {
    'model__alpha':np.logspace(-3, 3, 13)
}
search_a = GridSearchCV(
    ridge_pipe_a,
    param_grid=ridge_grid,
    scoring='neg_root_mean_squared_error',
    cv=cv,
    n_jobs=-1,
    return_train_score=True
)
search_a.fit(X_train_a,y_train_a)
print('Best Parameters:',search_a.best_params_)
print('Best CV RMSE:',-search_a.best_score_)
best_ridge_a = search_a.best_estimator_


# **Step 15-Diagnose overfitting and underfitting -- Train-test performance gap fr the california datset**

# In[27]:


def split_metrics(model, X_train, y_train, X_test, y_test):
    rows = []
    for split_name,Xs,ys in[ ('Train',X_train,y_train),('Test',X_test,y_test)]:
        pred = model.predict(Xs) 
        rows.append({'Split':split_name,'MAE':mean_absolute_error(ys, pred),'RMSE':np.sqrt(mean_squared_error(ys, pred)),'R2': r2_score(ys, pred)})
    return pd.DataFrame(rows)
for name in ['Linear Regression', 'Gradient Boosting', 'Ridge', 'Random Forest']:
    print(name)
    print(split_metrics(fitted_models_c[name],X_train_c,y_train_c,X_test_c,y_test_c))


# **Step 15-Diagnose overfitting and underfitting -- Train-test performance gap fr the ames datset**

# In[28]:


def split_metrics(model, X_train, y_train, X_test, y_test):
    rows = []
    for split_name,Xs,ys in[ ('Train',X_train,y_train),('Test',X_test,y_test)]:
        pred = model.predict(Xs) 
        rows.append({'Split':split_name,'MAE':mean_absolute_error(ys, pred),'RMSE':np.sqrt(mean_squared_error(ys, pred)),'R2': r2_score(ys, pred)})
    return pd.DataFrame(rows)
for name in ['Linear Regression', 'Gradient Boosting','Ridge','Random Forest']:
    print(name)
    print(split_metrics(fitted_models_a[name],X_train_a,y_train_a,X_test_a,y_test_a))


# **Step 16-Performing residual analysis fr the california dataset**

# In[29]:


selected_name_c =results_df_c.iloc[0]['Model']
y_pred_c =predictions_c[selected_name_c]
residuals_c = y_test_c.to_numpy()-y_pred_c
fig, ax = plt.subplots(figsize=(6.5,5))
ax.scatter(y_pred_c, residuals_c,alpha=0.65)
ax.axhline(0, linestyle='--')
ax.set_xlabel('Predicted value')
ax.set_ylabel('Residual =Actual-Predicted')
ax.set_title(f'Residuals vs Fitted :{selected_name_c}')
plt.show()
fig, ax = plt.subplots(figsize=(6.5,5))
sns.histplot(residuals_c,kde=True,ax=ax)
ax.set_title('Residual Distribution')
plt.show()
fig, ax = plt.subplots(figsize=(6.5,5))
ax.scatter(y_test_c, y_pred_c,alpha=0.65)
lo = min(y_test_c.min(),y_pred_c.min())
hi = max(y_test_c.max(),y_pred_c.max())
ax.plot([lo, hi],[lo, hi],linestyle='--')
ax.set_xlabel('Actual value')
ax.set_ylabel('Predicted value')
ax.set_title('Actual vs Predicted')
plt.show()


# **Step 16-Performing residual analysis fr the ames dataset**

# In[30]:


selected_name_a =results_df_a.iloc[0]['Model']
y_pred_a =predictions_a[selected_name_a]
residuals_a = y_test_a.to_numpy()-y_pred_a
fig,ax = plt.subplots(figsize=(6.5,5))
ax.scatter(y_pred_a,residuals_a,alpha=0.65)
ax.axhline(0,linestyle='--')
ax.set_xlabel('Predicted value')
ax.set_ylabel('Residual =Actual-Predicted')
ax.set_title(f'Residuals vs Fitted :{selected_name_c}')
plt.show()
fig,ax = plt.subplots(figsize=(6.5,5))
sns.histplot(residuals_a,kde=True,ax=ax)
ax.set_title('Residual Distribution')
plt.show()
fig,ax = plt.subplots(figsize=(6.5,5))
ax.scatter(y_test_a,y_pred_a,alpha=0.65)
lo = min(y_test_a.min(),y_pred_a.min())
hi = max(y_test_a.max(),y_pred_a.max())
ax.plot([lo, hi],[lo, hi],linestyle='--')
ax.set_xlabel('Actual value')
ax.set_ylabel('Predicted value')
ax.set_title('Actual vs Predicted')
plt.show()


# **Step 17-Inspect linear coefficients - Coefficient table(califronia dataset)**

# In[31]:


linear_fitted_c=fitted_models_c['Linear Regression']
feature_names_c=linear_fitted_c.named_steps['preprocess'].get_feature_names_out()
coefficients_c=linear_fitted_c.named_steps['model'].coef_
coef_df_c=pd.DataFrame({
    'Feature':feature_names_c,
    'Coefficient':coefficients_c,
    'AbsCoefficient':np.abs(coefficients_c)
}).sort_values('AbsCoefficient',ascending=False)
print(coef_df_c.head(20))


# In[ ]:





# **Step 17-Inspect linear coefficients - Coefficient table(ames dataset)**

# In[32]:


linear_fitted_a=fitted_models_a['Linear Regression']
feature_names_a=linear_fitted_a.named_steps['preprocess'].get_feature_names_out()
coefficients_a=linear_fitted_a.named_steps['model'].coef_
coef_df_a=pd.DataFrame({
    'Feature':feature_names_a,
    'Coefficient':coefficients_a,
    'AbsCoefficient':np.abs(coefficients_a)
}).sort_values('AbsCoefficient',ascending=False)
print(coef_df_a.head(20))


# **Step 20-Saving the selected pipeline and evidence(California dataset)**

# In[33]:


import json
import joblib
final_model_c=best_ridge_c
joblib.dump(final_model_c,'24MID0009_Lab01_california_Model.joblib')
results_df_c.to_csv('california_model_comparison.csv',index=False)
cv_results_c.to_csv('24MID0009_Lab01_california_results.csv', index=False)
run_metadata_c = {
    'random_seed':SEED,
    'target':target_c,
    'train_rows':len(X_train_c),
    'test_rows':len(X_test_c),
    'selected_model':'Tuned Ridge Regression'
}
with open('california_metadata.json','w') as f:
    json.dump(run_metadata_c,f,indent=2)


# **Step 20-Saving the selected pipeline and evidence(Ames dataset)**

# In[34]:


import json
import joblib
final_model_a=best_ridge_a
joblib.dump(final_model_a,'24MID0009_Lab01_ames_Model.joblib')
results_df_a.to_csv('ames_model_comparison.csv',index=False)
cv_results_a.to_csv('24MID0009_Lab01_ames_results.csv', index=False)
run_metadata_a = {
    'random_seed':SEED,
    'target':target_a,
    'train_rows':len(X_train_a),
    'test_rows':len(X_test_a),
    'selected_model':'Tuned Ridge Regression'
}
with open('ames_metadata.json','w') as f:
    json.dump(run_metadata_c,f,indent=2)


# In[35]:


print(results_df_c)


# In[36]:


print(results_df_a)


# **Final Summary**
# 1. Built regression models for the California Housing and Ames Housing datasets
# 2. Compared Linear Regression,Ridge,Elastic Net,Random Forest and the Gradient Boosting
# 3. On comparing,Gradient Boosting and Random Forest achieved the best prediction performance
# 4. Hyperparameter tuning using Grid Search(GridSearchCV) improved the Ridge model
# 5. Cross validation confirmed that the selected models generalize well
# 
# **Limitations**
# 
# 1. Performance depends on the quality and completeness of the datasets that we are taking
# 2. Extreme house prices may still produce larger prediction errors 
# 3. Only regression algorithms included in this lab were evaluated
# 4. Additional feature engineering and more data could further improve performance

# In[ ]:




