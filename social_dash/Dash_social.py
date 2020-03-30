# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 16:11:18 2018

@author: Jie.Hu
"""

import plotly
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table_experiments as dt
import dash_auth
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from keras.models import model_from_json

# read data
df_train = pd.read_csv('rsmv.csv')

df_test = pd.read_csv('eg2.csv')

# duplicate the df_train
df_dup = pd.concat([df_train]*df_test.shape[0], ignore_index=True)

# get corr
df = pd.concat([df_test, df_train], ignore_index=True)

df_corr = df.iloc[:,2:6].T.corr()

df_corr_test = df_corr.iloc[df_train.shape[0]:, 0:df_train.shape[0]].values.T.tolist()

corr_list = [item for sublist in df_corr_test for item in sublist]

# get mvname
mvname = df_test.iloc[:,0].T.tolist()

mvname_list = [item for item in mvname for i in range(df_train.shape[0])]

df_dup['MV'] = mvname_list

df_dup['Similarity'] = corr_list

# genre
df_dup['Genre'].value_counts(ascending=False)

df_dup['Genre'] = df_dup['Genre'].map({
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

# APP
app = dash.Dash()

# dropdown_1
mvname_options = df_dup['MV'].unique()
# dropdown_2
genre_options = df_dup['Genre'].unique()

app.layout = html.Div([
        html.Div([      
         html.Div([       
             html.Label('MVNAME'),    
            dcc.Dropdown(
                id='mvname-picker',
                options=[{'label': i, 'value': i} for i in mvname_options],
                value='United Kingdom'
               # multi=True
            )],
                style={'width': '40%', 'display': 'inline-block'}),           
       
         html.Div([  
           html.Label('Genre'),    
           dcc.Dropdown(
                id='genre-picker',
                options=[{'label': i, 'value': i} for i in genre_options],
                value='ALL'
            )],
                style={'width': '40%', 'display': 'inline-block'})],
                   
               style={
                 'borderBottom': 'thin lightgrey solid',
                 'backgroundColor': 'rgb(255,255,255)',
                 'padding': '10px 5px',
                 #'width': '100%',
                 'margin': '5px',
                 'font-family': 'Century Gothic'
                 }),

    html.H4('Gapminder DataTable'),
    dt.DataTable(
        rows=df_dup.to_dict('records'),

        # optional - sets the order of columns
        columns=sorted(df_dup.columns),

        row_selectable=True,
        filterable=True,
        sortable=True,
        selected_row_indices=[],
        id='datatable-gapminder'
    ),
    html.Div(id='selected-indexes'),
    dcc.Graph(
        id='graph-gapminder'
    ),
], className="container")


@app.callback(
    Output('datatable-gapminder', 'selected_row_indices'),
    [Input('graph-gapminder', 'clickData')],
    [State('datatable-gapminder', 'selected_row_indices')])
def update_selected_row_indices(clickData, selected_row_indices):
    if clickData:
        for point in clickData['points']:
            if point['pointNumber'] in selected_row_indices:
                selected_row_indices.remove(point['pointNumber'])
            else:
                selected_row_indices.append(point['pointNumber'])
    return selected_row_indices


@app.callback(
    Output('graph-gapminder', 'figure'),
    [Input('datatable-gapminder', 'rows'),
     Input('datatable-gapminder', 'selected_row_indices')])
def update_figure(rows, selected_row_indices):
    dff = pd.DataFrame(rows)
    fig = plotly.tools.make_subplots(
        rows=3, cols=1,
        subplot_titles=('Life Expectancy', 'GDP Per Capita', 'Population',),
        shared_xaxes=True)
    marker = {'color': ['#0074D9']*len(dff)}
    for i in (selected_row_indices or []):
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
        'y': dff['TA_T'],
        'type': 'bar',
        'marker': marker
    }, 3, 1)
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


#app.css.append_css({
#    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
#})

if __name__ == '__main__':
    app.run_server()
