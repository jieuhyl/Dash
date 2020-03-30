# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 16:38:52 2018

@author: Jie.Hu
"""
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 15:41:14 2018

@author: Jie.Hu
"""

import warnings
warnings.filterwarnings('ignore') 
import plotly
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table_experiments as dt
import dash_auth
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
from os import path
from wordcloud import WordCloud
import base64

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
#from sklearn.preprocessing import StandardScaler
#from keras.models import model_from_json

# read data
df_train = pd.read_csv('df_train.csv')

df_test = pd.read_csv('df_test.csv')
# duplicate the df_train
#df_dup = pd.concat([df_train]*df_test.shape[0], ignore_index=True)

# get whole data
df = pd.concat([df_test, df_train], ignore_index=True)

# missing
df.isnull().values.sum()
missing_ratio = df.isnull().sum() / len(df)
missing_ratio.sort_values(ascending=False)[:10]

# drop some features
drop_feat = ['Rating', 
             'DTR', 
             'Day of Week', 
             'Organic Conversation',
             'Humor',
             'Group Dynamic',
             'Paid Push',
             'Wikipedia Desktop']

df.drop(drop_feat, inplace = True, axis = 1)

# drop missing rows
df.dropna(subset=['Organic Intent',
                  'Twitter Mentions',
                  'Facebook PTAT',
                  'Wikipedia All',
                  'YouTube Views'], thresh=5, inplace=True)

# holiday
df['Holiday'].fillna('Normal', inplace=True)
df['Holiday'].value_counts()

def f(row):
    if row['Holiday'] == 'Normal':
        val = 0
    else:
        val = 1
    return val

df['Holiday'] = df.apply(f, axis=1)

# for year and month
df['Release Date'] = pd.to_datetime(df['Release Date'])
df['Year'] = df['Release Date'].dt.year
df['Month'] = df['Release Date'].dt.month.astype(str)

t_dummies1 = pd.get_dummies(df['Month'], prefix='Month')
df = df.join(t_dummies1)

# for genre
t_dummies2  = pd.get_dummies(df['Genre'], prefix='Genre')
df = df.join(t_dummies2)

# studio
# create studio counts
df['Studio_counts'] = df.groupby(['Studio'])['Title'].transform('count')
# create a group 
def f(row):
    if row['Studio_counts'] >= 20:
        val = "Large"
    elif row['Studio_counts'] >= 10 and row['Studio_counts'] < 20:
        val = "Medium"
    else:
        val = "Small"
    return val

df['Studio_size'] = df.apply(f, axis=1)

t_dummies3 = pd.get_dummies(df['Studio_size'], prefix='Studio')
df = df.join(t_dummies3)

drop_feat = ['Year', 
             'Month', 
             'Studio_counts',
             'Studio_size']

df.drop(drop_feat, inplace = True, axis = 1)

# get mtx
y = np.log1p(df['OBO']).values
X = df.iloc[:,5:].values

X_test , X_train =  X[:len(df_test), :], X[len(df_test):, :]
y_train = y[len(df_test):, ]

# modeling
mod_rf = RandomForestRegressor(max_depth = 7, 
                               max_features = 'auto', 
                               min_samples_leaf = 1, 
                               min_samples_split = 4,
                               n_estimators= 50,
                               random_state= 1337)
mod_rf.fit(X_train, y_train)

y_pred = mod_rf.predict(X_test)

df['OBO'].loc[df['OBO'].isnull()] = np.expm1(y_pred)

df.isnull().values.sum()

# add prediction and comment
mapping_title = {'First Man': 'The predction is $23M, and why', 
                    'Bad Times at the El Royale':'The predction is $15M, and why', 
                    'Goosebumps 2: Haunted Halloween':'The predction is $10M, and why'}
df['Comment'] = df['Title'].map(mapping_title)



# APP =========================================================================
app = dash.Dash()

colors = {
    'background': '#111111',
    'text': '#808080'
}

available_indicators = np.array(['OBO', 'UA_T', 'TA_T', 'DI_T', 'FC_T'])

# dropdown_1
title_options = df_test['Title'].unique()
# dropdown_2
genre_options = df['Genre'].unique()

hover_text = []

for index, row in df.iterrows():
    hover_text.append(('Title: {mvname}<br>'+
                       'Genre: {genre}<br>'+
                       'Organic Intent: {oi:,.0f}<br>'+
                       'Twitter: {tm:,.0f}<br>'+
                       'Facebook: {fb:,.0f}<br>'+
                       'Wikipedia: {wi:,.0f}<br>'+
                       'YouTube Views: {yo:,.0f}<br>'+
                       'OBO: ${value:,.0f}').format(mvname=row['Title'],
                                                    genre=row['Genre'],
                                                    oi=row['Organic Intent'],
                                                    tm=row['Twitter Mentions'],
                                                    fb=row['Facebook PTAT'],
                                                    wi=row['Wikipedia All'],
                                                    yo=row['YouTube Views'],
                                                    value=row['OBO']))

df['Text'] = hover_text

k = df_test.shape[0]
df_k = df.iloc[:k,]
df_kk = df.iloc[k:,]

markdown_text ='''
**Method:** Artificial Neural Network. This may not be the best method here since data is too 'ez' for the network
which means the sample size is really samll and only three attributes are considered. The MAPE is around 25%.  
**Graph:** Implement 4D data to 2D graph. X-axis is the value for the reserved seat, Y-axis is actual value for OBO, bubble size is
proportional to theater count, bubble color reflects the rt score, you can you RHS colorscale bar as a reference.  
**Correlation:** The correlations of reserved seat at Wednesday afternoon, Wednesday night and
Thursday afternoon to OBO are 88%, 88% and 93% respectively. We are using Wednesday afternoon data.  
**Keep in Mind:** This is not an official forecast report but more like a tool to play with!
'''


app.layout = html.Div(children=[
    html.H1(
        children='Social Dash',
        style={
            'textAlign': 'center',
            'color': colors['text'],
            'margin': '5px',
            'font-family': 'Century Gothic'
        }
    ),

    html.Div(children='12-10-2018', style={
        'textAlign': 'center',
        'color': colors['text'],
        'margin': '5px',
        'font-family': 'Century Gothic'
    }),

    #dcc.Markdown(children = markdown_text),
          
    html.Div([       
             html.Label('Title'),    
            dcc.Dropdown(
                id='title-picker',
                options=[{'label': i, 'value': i} for i in title_options],
                value=title_options[0],
                multi=False
            )],
                style={'width': '50%', 'display': 'inline-block'}),  
    
    html.Div([  
           html.Label('Genre'),    
           dcc.Dropdown(
                id='genre-picker',
                options=[{'label': i, 'value': i} for i in genre_options],
                value=[genre_options[0]],
                multi=True
            )],
                style={'width': '50%', 'display': 'inline-block'}),
        
      html.Div(id='result',
                  style={
                 #'borderBottom': 'thin lightgrey solid',
                 #'backgroundColor': 'rgb(234,234,234)',
                 'padding': '15px 15px 15px 0px',
                 'font-weight': 'bold',
                 #'width': '100%',
                 #'margin': { 'l': 10, 'b': 20, 't': 30, 'r': 40},
                 'color': colors['text'],
                 'margin': '5px',
                 'font-family': 'Century Gothic'
                 }),


        dt.DataTable(
        rows=[{}],
        #rows=df.iloc[1:,:].to_dict('records'),

        # optional - sets the order of columns
        #columns=sorted(df_dup.columns),

        row_selectable=True,
        filterable=True,
        sortable=True,
        selected_row_indices=[],
        id='datatable'),
                    
      
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
    [Input(component_id='title-picker', component_property='value')])
def update_table(selected_title):
    filtered_df = df_k[df_k['Title']==selected_title]
    
    return filtered_df['Comment'] 
    
        
if __name__ == '__main__':
    app.run_server()