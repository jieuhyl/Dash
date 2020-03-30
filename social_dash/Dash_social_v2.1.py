# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 22:32:21 2018

@author: Jie.Hu
"""


import plotly
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table_experiments as dt
#import dash_auth
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import cosine_similarity
#from keras.models import model_from_json

# read data
df_train = pd.read_csv('rsmv.csv')

df_test = pd.read_csv('it.csv')

# duplicate the df_train
#df_dup = pd.concat([df_train]*df_test.shape[0], ignore_index=True)

# get corr
df = pd.concat([df_test, df_train], ignore_index=True)
'''
df_corr = df.iloc[:,2:6].T.corr()

corr_list = df_corr.iloc[:, 0].values.T.tolist()

#corr_list = [item for sublist in df_corr_test for item in sublist]

# get mvname
#mvname = df_test.iloc[:,0].T.tolist()

#mvname_list = [item for item in mvname for i in range(df_train.shape[0])]

#df_dup['MV'] = mvname_list

df['Similarity'] = corr_list
'''
# genre
df['Genre'].value_counts(ascending=False)

df['Genre'] = df['Genre'].map({
                              'ACTION DRAMA': 'ACTION',
                              'HUMN INTST DRMA': 'DRAMA',
                              'FAMILY':'FAMILY',
                              'ACTION ADVENTURE':'ACTION',
                              'HUMN INTST CMDY':'COMEDY',
                              'HORROR':'HORROR',
                              'SUSPNS/THRILLER': 'SUSPENSE',
                              'COMEDY': 'COMEDY',
                              'DRAMA': 'DRAMA',
                              'ZANY COMEDY':'COMEDY',
                              'ACTION COMEDY':'ACTION',
                              'ACTION':'ACTION',
                              'THRILLER':'SUSPENSE',
                              'ANIMATION': 'ANIMATION',
                              'ROMANTIC COMEDY':'COMEDY',
                              'ROMANTIC DRAMA': 'DRAMA',
                              'ROMANCE': 'DRAMA',
                              'DOCUMENTARY': 'DOCUMENTARY'
                               })
# prediction
X_train = df_train.iloc[:,2:6].values
y_train = df_train['OBO'].values

X_test = df_test.iloc[:,2:6].values 

mod_reg = LinearRegression()
mod_reg.fit(X_train, y_train)
y_pred = mod_reg.predict(X_test)

# APP
app = dash.Dash()

# dropdown_1
studio_options = df['Studio'].unique()
# dropdown_2
genre_options = df['Genre'].unique()


app.layout = html.Div([
        
   html.H4('IT DataTable'),
   
    html.Div([      
         html.Div([       
             html.Label('Studio'),    
            dcc.Dropdown(
                id='studio-picker',
                options=[{'label': i, 'value': i} for i in studio_options],
                value=[studio_options[0],studio_options[1]],
                multi=True
            )],
                style={'width': '33%', 'display': 'inline-block'}),           
       
         html.Div([  
           html.Label('Genre'),    
           dcc.Dropdown(
                id='genre-picker',
                options=[{'label': i, 'value': i} for i in genre_options],
                value=[genre_options[0],genre_options[1]],
                multi=True
            )],
                style={'width': '33%', 'display': 'inline-block'})]),
           
    
    dt.DataTable(
        rows=[{}],
        #rows=df.iloc[1:,:].to_dict('records'),

        # optional - sets the order of columns
        #columns=sorted(df_dup.columns),

        row_selectable=True,
        filterable=True,
        sortable=True,
        selected_row_indices=[],
        id='datatable'
    ),
    
    dcc.Graph(
        id='graph'
    ),
], className="container")

    
#1    
@app.callback(Output('datatable', 'rows'), 
              [Input('studio-picker', 'value'),
               Input('genre-picker', 'value')])

def update_table(selected_studio, selected_genre):
    
 if selected_studio == []:
        studio_list = list(df['Studio'].unique())
 else:
        studio_list = selected_studio
 if selected_genre == []:
        genre_list = list(df['Genre'].unique())
 else:
        genre_list = selected_genre
         
 filtered_df_train = df_train[(df_train['Studio'].isin(studio_list)) & (df_train['Genre'].isin(genre_list))]
    
 filtered_df = pd.concat([df_test, filtered_df_train], ignore_index=True) 
     
 X_total = filtered_df.iloc[:,2:6].values
 mm = MinMaxScaler()
 X_total = mm.fit_transform(X_total)
 gmm = GaussianMixture(n_components=4, covariance_type='full', random_state=1337).fit(X_total)
 filtered_df['Class'] = gmm.predict(X_total)

 filtered_df['Similarity_1'] = cosine_similarity(gmm.predict_proba(X_total)[0:1,:], 
                                        gmm.predict_proba(X_total))[0]

 filtered_df['Similarity_2'] = cosine_similarity(X_total[0:1,:], 
                                       X_total)[0]

 filtered_df['Similarity'] = (filtered_df['Similarity_1']+filtered_df['Similarity_2'])/2
 filtered_df.drop(['Similarity_1', 'Similarity_2'], axis=1, inplace=True)  
           
 return filtered_df.to_dict('records')
    
      

#2
@app.callback(
    Output('datatable', 'selected_row_indices'),
    [Input('graph', 'clickData')],
    [State('datatable', 'selected_row_indices')])
def update_selected_row_indices(clickData, selected_row_indices):
    if clickData:
        for point in clickData['points']:
            if point['pointNumber'] in selected_row_indices:
                selected_row_indices.remove(point['pointNumber'])
            else:
                selected_row_indices.append(point['pointNumber'])      
    return selected_row_indices + [0]

#3
@app.callback(
    Output('graph', 'figure'),
    [Input('datatable', 'rows'),
     Input('datatable', 'selected_row_indices')])
    
def update_figure(rows, selected_row_indices):
    #dff = pd.concat([df.iloc[0:1,:], pd.DataFrame(rows, columns=df.columns)])
    dff = pd.DataFrame(rows, columns=df.columns)
    
    fig = plotly.tools.make_subplots(
        rows=4, cols=1,
        subplot_titles=('UA_T', 'TA_T', 'DI_T', 'FC_T'),
        shared_xaxes=True, print_grid=False)
    marker = {'color': ['#0074D9']*len(dff)}
    for i in (selected_row_indices+[0] or []):
        marker['color'][i] = '#FF851B'
    fig.append_trace({
        'x': dff['MOVIE'],
        'y': dff['UA_T'],
        'type': 'bar',
        'marker': marker
    }, 1, 1)
    fig.append_trace({
        'x': dff['MOVIE'],
        'y': dff['TA_T'],
        'type': 'bar',
        'marker': marker
    }, 2, 1)
    fig.append_trace({
        'x': dff['MOVIE'],
        'y': dff['DI_T'],
        'type': 'bar',
        'marker': marker
    }, 3, 1)
    fig.append_trace({
        'x': dff['MOVIE'],
        'y': dff['FC_T'],
        'type': 'bar',
        'marker': marker
    }, 4, 1)
    fig['layout']['showlegend'] = False
    fig['layout']['height'] = 800
    fig['layout']['margin'] = {
        'l': 40,
        'r': 10,
        't': 60,
        'b': 200
    }
    fig['layout']['yaxis3']['type'] = 'log'
    return fig



if __name__ == '__main__':
    app.run_server()
