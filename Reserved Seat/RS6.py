# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 16:18:42 2018

@author: Jie.Hu
"""

import warnings
warnings.filterwarnings('ignore') 
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_auth
from dash.dependencies import Input, Output
import plotly.graph_objs as go

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
from scipy import stats
from scipy.stats import norm, skew 


df = pd.read_csv('RS_nrg.csv', encoding='cp1252')

'''
# check missing
df.isnull().values.sum()
missing_ratio = df.isnull().sum() / len(df)
'''
df.columns
d = {'TOTAL':'RS_TOTAL', 
     'RESERVED':'RS', 
     'AVAILABLE':'RS_AVAILABLE', 
     'Opening Weekend B.O.':'OBO',
     'Opening Theater Count':'TC',
     'Critic Score': 'RT'}
df_new=df.groupby(['NAME', 'RUNNAME'], as_index=False).agg({'TOTAL':'sum', 
                               'RESERVED':'sum',
                               'AVAILABLE':'sum',
                               'Opening Weekend B.O.':'mean',
                               'Opening Theater Count':'mean',
                               'Critic Score':'mean'}).rename(columns=d)

df_new_rs = df_new.pivot(index = 'NAME', columns= 'RUNNAME', values = 'RS').reset_index()

df_new=df.groupby(['NAME'], as_index=False).agg({'TOTAL':'sum', 
                               'RESERVED':'sum',
                               'AVAILABLE':'sum',
                               'Opening Weekend B.O.':'mean',
                               'Opening Theater Count':'mean',
                               'Critic Score':'mean'}).rename(columns=d)
#df_new = pd.concat([df_new, df_new_rs], axis = 1)
df_new['RS_W_A'] = df_new_rs['Noon Day Before Pre-Shows']
df_new['RS_W_N'] = df_new_rs['Late Night Before Pre-Shows']
df_new['RS_T_A'] = df_new_rs['Afternoon Before Pre-Shows']

# check missing
df_new.isnull().values.sum()
missing_ratio = df_new.isnull().sum() / len(df_new)
df_new.loc[df_new['RT']<=5, 'RT'] = 5
df_new.to_csv('C:/Users/Jie.Hu/Desktop/Reserved Seat/df_new.csv', index=False) 

df_new['RS_W_N'][df_new['NAME'] == 'Just Getting Started'] = 0
df_new['RS_T_A'][df_new['NAME'] == 'Just Getting Started'] = 0
df_new['RS_T_A'][df_new['NAME'] == 'Teen Titans Go! To the Movies'] = 1300
df_new['RS'][df_new['NAME'] == 'Teen Titans Go! To the Movies'] = 2580
df_new['RS_T_A'][df_new['NAME'] == 'Mission: Impossible - Fallout'] = 13000
df_new['RS'][df_new['NAME'] == 'Mission: Impossible - Fallout'] = 25357

'''
from collections import Counter

# Outlier detection 
def detect_outliers(df,n,features):
    """
    Takes a dataframe df of features and returns a list of the indices
    corresponding to the observations containing more than n outliers according
    to the Tukey method.
    """
    outlier_indices = []
    
    # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col], 75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1
        
        # outlier step
        outlier_step = 1.5 * IQR
        
        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index
        
        # append the found outlier indices for col to the list of outlier indices 
        outlier_indices.extend(outlier_list_col)
        
    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)        
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
    
    return multiple_outliers   

# detect outliers from 
Outliers_to_drop = detect_outliers(df_new, 1, ['RS_W_A', 'OBO'])

# show the outliers
df_new_drop = df_new.loc[Outliers_to_drop] 
'''
# drop the outliers big three movies
#df = df.drop(Outliers_to_drop, axis = 0, inplace = True)
# 11,15,108,0,5,32,70 
df_new = df_new.drop(df.index[[11,15,108]])

#RT for less than 5 to 5
df_new.loc[df_new['RT']<=5, 'RT'] = 5
df_new.loc[df_new['RS_W_A']<=10, 'RS_W_A'] = 10

#df_new.to_csv('C:/Users/Jie.Hu/Desktop/Reserved Seat/df_new.csv', index=False) 



'''
# check response
df_new['OBO'].describe()

# orginal data
sns.distplot(df_new['RT'] , fit=norm);
# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(df_new['RT'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(df_new['RT'], plot=plt)
plt.show()

# log transformation, NEED (OBO, RS)
sns.distplot(np.log(df_new['RS_W_A']), fit=norm);
# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(np.log(df_new['RS_W_A']))
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(np.log(df_new['RS_W_A']), plot=plt)
plt.show()


# correlations
corrmat = df_new.corr()
sns.heatmap(corrmat, vmax=.8, annot=True, cmap="RdYlGn")

top_corr_features = corrmat.index[abs(corrmat['OBO']) >= 0.5]
sns.heatmap(df_new[top_corr_features].corr(), annot=True, cmap="RdYlGn")

# check missing
df_new.isnull().values.sum()
'''

''' Modeling '''
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import optimizers
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV


# get mtx
y = df_new['OBO'].values
df_new['RS_W_A_log'] = np.log(df_new['RS_W_A'])
X = df_new.iloc[:, [10,5,6]].values

# Feature Scaling MinMaxScaler()
sc = StandardScaler()
X = sc.fit_transform(X)
y = np.log(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1337)


'''
from sklearn.cluster import DBSCAN
# Compute DBSCAN
db = DBSCAN(eps=1, min_samples=10).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

print(df_new[labels == -1])
'''
# , metrics = ['mape']
def build_reg(optimizer):
    reg = Sequential()
    reg.add(Dense(units = 100, activation = 'relu', input_dim = 3))
    reg.add(Dropout(p = 0.01))
    reg.add(Dense(units = 50, activation = 'relu'))
    reg.add(Dropout(p = 0.01))
    reg.add(Dense(units = 10, activation = 'relu'))
    reg.add(Dropout(p = 0.01))
    reg.add(Dense(units = 1))
    #sgd = optimizers.SGD(lr=0.01, decay=0.001, momentum=0.9, nesterov=True)
    reg.compile(optimizer = optimizer, loss = 'mse', metrics = ['mape'])
    return reg

reg = KerasRegressor(build_fn = build_reg, shuffle = True)
# 'adam', 'rmsprop', 
parameters = {'batch_size': [2,4,8,16],
              'epochs': [500, 1000],
              'optimizer': ['sgd']
              }
grid_search = GridSearchCV(estimator = reg,
                           param_grid = parameters,
                           scoring = 'neg_mean_squared_error',
                           cv = 5)
grid_search = grid_search.fit(X, y)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_


# last step
import tensorflow as tf
import random as rn
import os
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(1)
rn.seed(1)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
from keras import backend as K
tf.set_random_seed(1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

def build_reg():
    reg = Sequential()
    reg.add(Dense(units = 30, activation = 'relu', input_dim = 3))
    reg.add(Dropout(p = 0.01))
    reg.add(Dense(units = 20, activation = 'relu'))
    reg.add(Dropout(p = 0.01))
    reg.add(Dense(units = 10, activation = 'relu'))
    reg.add(Dropout(p = 0.01))
    reg.add(Dense(units = 1))
    reg.compile(optimizer = 'sgd', loss = 'mse', metrics = ['mape'])
    return reg

# , batch_size = 5, epochs = 200
reg = KerasRegressor(build_fn = build_reg , shuffle = True, batch_size = 2, epochs = 500)

# Fitting the ANN to the Training set
reg.fit(X, y)


# Predicting the Test set results
y_pred = reg.predict(X)

# MAPE
np.mean(np.abs(np.exp(y_pred)-np.exp(y))/np.exp(y))


# Regression
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

# Define error measure for official scoring : RMSE
def rmse_cv_train(model):
    rmse= np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv = 10))
    return(rmse)
    
# 1 Ridge
ridge = RidgeCV(alphas = [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60])
ridge.fit(X, y)
alpha = ridge.alpha_
print("Best alpha :", alpha)

print("Try again for more precision with alphas centered around " + str(alpha))
ridge = RidgeCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85, 
                          alpha * .9, alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15,
                          alpha * 1.25, alpha * 1.3, alpha * 1.35, alpha * 1.4], 
                cv = 10)
ridge.fit(X, y)
alpha = ridge.alpha_
print("Best alpha :", alpha)

print("Ridge RMSE on Training set :", rmse_cv_train(ridge).mean())
#print("Ridge RMSE on Test set :", rmse_cv_test(ridge).mean())
y_train_rdg = ridge.predict(X)
#y_test_rdg = ridge.predict(X_test)

# Plot residuals
plt.scatter(y_train_rdg, y_train_rdg - y, c = "blue", marker = "s", label = "Training data")
#plt.scatter(y_test_rdg, y_test_rdg - y_test, c = "lightgreen", marker = "s", label = "Validation data")
plt.title("Linear regression with Ridge regularization")
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.legend(loc = "upper left")
plt.hlines(y = 0, xmin = 12, xmax = 20, color = "red")
plt.show()

# Plot predictions
plt.scatter(y_train_rdg, y, c = "blue", marker = "s", label = "Training data")
#plt.scatter(y_test_rdg, y_test, c = "lightgreen", marker = "s", label = "Validation data")
plt.title("Linear regression with Ridge regularization")
plt.xlabel("Predicted values")
plt.ylabel("Real values")
plt.legend(loc = "upper left")
plt.plot([12, 20], [12, 20], c = "red")
plt.show()

# Plot important coefficients
coefs = pd.Series(ridge.coef_, index = df_new.iloc[:, [10,5,6]].columns)
print("Ridge picked " + str(sum(coefs != 0)) + " features and eliminated the other " +  \
      str(sum(coefs == 0)) + " features")
imp_coefs = pd.concat([coefs.sort_values(ascending=True).head(10),
                     coefs.sort_values(ascending=True).tail(10)])
imp_coefs.plot(kind = "barh")
plt.title("Coefficients in the Ridge Model")
plt.show()


# MADMEAN
np.mean(np.abs(np.exp(y_train_rdg)-np.exp(y)))/np.mean(np.exp(y)) * 100



''' gradient boosting '''
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
mod_gb = GradientBoostingRegressor(random_state= 1337)
mod_gb.fit(X, y)

# Predicting the Test set results
y_pred = mod_gb.predict(X)
mape_gb = np.mean(np.abs((np.exp(y) - np.exp(y_pred)) / np.exp(y))) * 100

#Print model report:
print ("\nModel Report")
print ("MAPE : %.2f" % mape_gb)



parameters = {'learning_rate':[0.1, 0.2], 
              'max_depth':[4,6,8,10],
              'max_features':[2,3],
              'min_samples_leaf':[1,3,5,7,9],
              'min_samples_split':[2,4,6,8,10],
              'n_estimators':[80,100,120],
              'subsample':[0.8,0.9]}
                                       
grid_search = GridSearchCV(estimator = mod_gb,
                           param_grid = parameters,
                           scoring='neg_mean_squared_error',
                           cv = 5,
                           n_jobs = -1)
grid_search = grid_search.fit(X, y)
grid_search.best_params_, grid_search.best_score_

# last step
mod_gb = GradientBoostingRegressor(learning_rate=0.2, 
                                        max_depth = 6, 
                                        max_features = 2,
                                        min_samples_leaf = 3, 
                                        min_samples_split = 2,
                                        n_estimators=80, 
                                        subsample=0.8,
                                        random_state= 1337 )
mod_gb.fit(X, y)

# Predicting the Test set results
y_pred = mod_gb.predict(X)

# check r2 and mape
mape_gb = np.mean(np.abs((np.exp(y) - np.exp(y_pred)) / np.exp(y))) * 100
r2_gb = r2_score(np.exp(y), np.exp(y_pred))

#Print model report:
print ("\nModel Report")
print ("R2 Score: %.2f" % r2_gb)
print ("MAPE Score: %.2f" % mape_gb)


''' random forest '''
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
mod_rf = RandomForestRegressor(random_state=1337)
mod_rf.fit(X_train, y_train)
y_pred = mod_rf.predict(X_train)

mape_rf = np.mean(np.abs((np.exp(y_train) - np.exp(y_pred)) / np.exp(y_train))) * 100

# k fold and grid
parameters = {
              'bootstrap': [True, False],
              'max_depth':[6,8,10,None],
              'max_features':['auto', 'sqrt'],
              'min_samples_leaf':[1,3,5],
              'min_samples_split':[2,4,6],             
              'n_estimators':[100,200]
              }
                                       
grid_search = GridSearchCV(estimator = mod_rf,
                           param_grid = parameters,
                           scoring= 'neg_mean_squared_error',
                           cv = 5,
                           n_jobs = -1)
grid_search = grid_search.fit(X, y)
grid_search.best_score_,  grid_search.best_params_


# last step
mod_rf = RandomForestRegressor(bootstrap = False,
                               max_depth = 6, 
                               max_features = 'sqrt', 
                               min_samples_leaf = 3, 
                               min_samples_split = 2,
                               n_estimators=200,
                               random_state= 1337)
mod_rf.fit(X,y)

# Predicting the Test set results
y_pred = mod_rf.predict(X)
# check r2 and mape
mape_rf = np.mean(np.abs((np.exp(y) - np.exp(y_pred)) / np.exp(y))) * 100
r2__rf = r2_score(np.exp(y_test), np.exp(y_pred))



#Print model report:
print ("\nModel Report")
print ("R2 Score: %.2f" % r2__rf)
print ("MAPE Score: %.2f" % mape_rf)

# feature importance
fi = mod_rf.feature_importances_
predictors = [x for x in bo.columns if x not in ['OBO']]
feat_imp = pd.Series(mod_rf.feature_importances_, predictors).sort_values(ascending=False)
feat_imp.plot(kind='bar', title='Feature Importances')
plt.ylabel('Feature Importance Score')


#==============================================================================
user_password = [['username', 'password'], ['nrgms', '1234']]

app = dash.Dash()

auth = dash_auth.BasicAuth(app, user_password)

server = app.server

colors = {
    'background': '#111111',
    'text': '#808080'
}


hover_text = []

for index, row in df_new.iterrows():
    hover_text.append(('Movie Name: {mvname}<br>'+
                       'Reserved_Total: {rs:,.0f}<br>'+
                       'Reserved_Wed_Afternoon: {rswa:,.0f}<br>'+
                       'Theater Count: {tc:,.0f}<br>'+
                       'Rotten Tomato: {rt}<br>'+
                       'OBO: ${value:,.0f}').format(mvname=row['NAME'],
                                                    rs=row['RS'],
                                                    rswa=row['RS_W_A'],
                                                    tc=row['TC'],
                                                    rt=row['RT'],
                                                    value=row['OBO']))

df_new['Text'] = hover_text


markdown_text ='''
**Method:** Artificial Neural Network, this is not a linera model like theatrical model or social model.
So if you scale one attribute up while holding the rest, the prediction may not be necessary go up.  
**Graph:** Implement 4D data to 2D graph. X-axis is the value for the reserved seat, Y-axis is actual value for OBO, bubble size is
proportional to theater count, bubble color reflects the rt score, you can you RHS colorscale bar as a reference.  
**Correlation:** The correlations of reserved seat at Wednesday afternoon, Wednesday night and
Thursday afternoon to OBO are 88%, 88% and 93% respectively. We are using Wednesday afternoon data.  
**Keep in Mind:** This is not an official forecast report but more like a tool to play with! 
'''

'''
### Dash apps can be written in Markdown.
Dash uses the [CommonMark](http://commonmark.org/)
specification of Markdown.\
Check out their [60 Second Markdown Tutorial](http://commonmark.org/help/)
if this is your first introduction to Markdown!
'''


app.layout = html.Div(children=[
    html.H1(
        children='AI meets Reserved Seat',
        style={
            'textAlign': 'center',
            'color': colors['text'],
            'margin': '5px',
            'font-family': 'Century Gothic'
        }
    ),

    html.Div(children='NRG, Marketing Science', style={
        'textAlign': 'center',
        'color': colors['text'],
        'margin': '5px',
        'font-family': 'Century Gothic'
    }),
     
    dcc.Markdown(children = markdown_text),

     html.Div([
        html.Div([       
             html.Label('Reserved Seat:'),    
              dcc.Input(id='seat', placeholder='Enter a value...', type='text')],
              style={'width': '20%', 'display': 'inline-block'}),           
       
        html.Div([  
           html.Label('Theater Count:'),    
           dcc.Input(id='screen', placeholder='Enter a value...', type='text')],
           style={'width': '20%', 'display': 'inline-block'}),
                   
        html.Div([  
           html.Label('Rotten Tomato:'),            
           dcc.Input(id='rt', placeholder='Enter a value...', type='text')],
           style={'width': '20%', 'display': 'inline-block'}),
                 
         
        html.Div(id='result',
                  style={
                 #'borderBottom': 'thin lightgrey solid',
                 #'backgroundColor': 'rgb(234,234,234)',
                 #'padding': '20px 10px',
                 #'width': '100%',
                 #'margin': '5px',
                 #'font-family': 'Century Gothic'
                 })],
                 style={#'margin': '5px',
                        'borderBottom': 'thin lightgrey solid',
                        'backgroundColor': 'rgb(234,234,234)',
                        'padding': '10px 10px',
                        'font-family': 'Century Gothic'}),
             

    dcc.Graph(
        id='graph',
        style={'height': 600}
    )
],
                 style={'font-family': 'Century Gothic',
                        'margin': '5px'
        })


#1
@app.callback(
     Output(component_id='result', component_property='children'),
    [Input(component_id='seat', component_property='value'),
     Input(component_id='screen', component_property='value'),
     Input(component_id='rt', component_property='value')])
def update_three_inputs(seat, screen, rt):
    if seat is not None and seat is not '' and screen is not None and screen is not '' and rt is not None and rt is not '':
        try:
            Prediction = np.exp(reg.predict(sc.transform([[np.log(float(seat)), float(screen), float(rt)]])).tolist())
           # new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
            return 'With {} reserved seat, {} theater count and {} rotten tomato, the prediction is ${:,.1f}M. And confidence interval is from ${:,.1f}M to ${:,.1f}M.'.\
                format(seat, screen, rt, Prediction/1000000, 0.8*Prediction/1000000, 1.2*Prediction/1000000, 2)
        except ValueError:
            return 'Unable to give the prediction'


#2
@app.callback(Output('graph', 'figure'),
              [Input('seat', 'value'),
               Input('screen', 'value'),
               Input('rt', 'value')])
    
def update_figure(selected_seat, selected_screen, selected_rt):
    Prediction = np.exp(reg.predict(sc.transform([[np.log(float(selected_seat)), float(selected_screen), float(selected_rt)]])).tolist())
    Predicted_row = ['New', 100, 100, 100, Prediction, float(selected_screen), float(selected_rt), float(selected_seat), 100, 100, 100, '']
    #df_new.loc['New Movie'] = Predicted_row
    df_new.loc[130] = Predicted_row

    
    return {
            'data': [
                go.Scatter(
                    x = df_new['RS_W_A'],
                    y = df_new['OBO'],
                    text = df_new['Text'],
                    hoverinfo = 'text',
                    mode='markers',
                    opacity=0.7,
                    marker={
                        'size': df_new['TC']/100,
                        'color': df_new['RT'],
                        'colorscale': 'Jet',
                        'showscale': True,
                        'line': {'width': 0.5, 'color': 'white'}
                    },
                )
            ],
            'layout': go.Layout(
                font=dict(family='Century Gothic'),
                title = 'Bubble Chart for Opening Box Office',
                xaxis={'title': 'Reserved Seat'},
                yaxis={'title': 'Opening Box Office'},
                #margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                hovermode='closest',
                annotations = [dict(x=selected_seat, y=Prediction, text = 'New Movie')]
            )
        } 
         
# 
if __name__ == '__main__':
    app.run_server(port=8585, host='10.16.21.147')
