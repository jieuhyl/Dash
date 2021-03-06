# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 16:33:41 2018

@author: Jie.Hu
"""


import warnings
warnings.filterwarnings('ignore') 
import dash
import dash_core_components as dcc
import dash_html_components as html
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
Outliers_to_drop = detect_outliers(df_new, 2, ['RS', 'OBO', 'TC', 'RT'])

# show the outliers
#df_new_drop = df_new.loc[Outliers_to_drop] 

# drop the outliers
#df = df.drop(Outliers_to_drop, axis = 0, inplace = True)
'''

#RT for less than 5 to 5
df_new.loc[df_new['RT']<=5, 'RT'] = 5
df_new.loc[df_new['RS_W_A']<=10, 'RS_W_A'] = 10
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
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# get mtx
y = df_new['OBO'].values
df_new['RS_W_A_log'] = np.log(df_new['RS_W_A'])
X = df_new.iloc[:, [10,5,6]].values

# Feature Scaling
sc = StandardScaler()
X = sc.fit_transform(X)
y = np.log(y)

'''#
def build_reg(optimizer):
    reg = Sequential()
    reg.add(Dense(units = 2, activation = 'relu', input_dim = 3))
    reg.add(Dropout(p = 0.01))
    #reg.add(Dense(units = 10, activation = 'relu'))
    #reg.add(Dropout(p = 0.01))
    reg.add(Dense(units = 1))
    reg.compile(optimizer = optimizer, loss = 'mse', metrics = ['mape'])
    return reg

reg = KerasRegressor(build_fn = build_reg)

parameters = {'batch_size': [5,10,15,20,30,40,50],
              'epochs': [100,200],
              'optimizer': ['rmsprop']}
grid_search = GridSearchCV(estimator = reg,
                           param_grid = parameters,
                           scoring = 'neg_mean_squared_error',
                           cv = 5)
grid_search = grid_search.fit(X, y)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
'''

# last step
def build_reg():
    reg = Sequential()
    reg.add(Dense(units = 2, activation = 'relu', input_dim = 3))
    reg.add(Dropout(p = 0.01))
    #reg.add(Dense(units = 2, activation = 'relu'))
    #reg.add(Dropout(p = 0.01))
    reg.add(Dense(units = 1))
    reg.compile(optimizer = 'rmsprop', loss = 'mse', metrics = ['mape'])
    return reg

# , batch_size = 5, epochs = 200
reg = KerasRegressor(build_fn = build_reg , batch_size = 5, epochs = 500)

# Fitting the ANN to the Training set
reg.fit(X, y)

'''
# Predicting the Test set results
y_pred = reg.predict(X)

# MAPE
np.mean(np.abs(np.exp(y_pred)-np.exp(y))/np.exp(y))
'''

#==============================================================================
app = dash.Dash()

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


markdown_text = '''
## NOT TITLE YET!!!'''

'''
### Dash apps can be written in Markdown.
Dash uses the [CommonMark](http://commonmark.org/)
specification of Markdown.\
Check out their [60 Second Markdown Tutorial](http://commonmark.org/help/)
if this is your first introduction to Markdown!
'''


app.layout = html.Div(children=[
    html.H1(
        children='National Research Group',
        style={
            'textAlign': 'center',
            'color': colors['text'],
            'margin': '5px',
            'font-family': 'Century Gothic'
        }
    ),

    html.Div(children='Marketing Science', style={
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
                 'borderBottom': 'thin lightgrey solid',
                 'backgroundColor': 'rgb(255,255,255)',
                 'padding': '20px 0px',
                 #'width': '100%',
                 #'margin': '5px',
                 #'font-family': 'Century Gothic'
                 })],
                 style={'margin': '5px',
                        'font-family': 'Century Gothic'}),
             

    dcc.Graph(
        id='graph',
        style={'height': 600}
    )
])


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
     
    #annotations = list([dict(text='Trace type:', x=0, y=1, yref='paper', align='left', showarrow=False)])

    #layout['annotations'] = annotations    
         

if __name__ == '__main__':
    app.run_server()
