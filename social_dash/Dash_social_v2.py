# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 11:25:05 2018

@author: Jie.Hu
"""

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

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import StandardScaler
from keras.models import model_from_json

# read data
df_train = pd.read_csv('rsmv.csv')

df_test = pd.read_csv('it.csv')

# duplicate the df_train
#df_dup = pd.concat([df_train]*df_test.shape[0], ignore_index=True)

# get corr
df = pd.concat([df_test, df_train], ignore_index=True)

df_corr = df.iloc[:,2:6].T.corr()

corr_list = df_corr.iloc[:, 0].values.T.tolist()

#corr_list = [item for sublist in df_corr_test for item in sublist]

# get mvname
#mvname = df_test.iloc[:,0].T.tolist()

#mvname_list = [item for item in mvname for i in range(df_train.shape[0])]

#df_dup['MV'] = mvname_list

df['Similarity'] = corr_list

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


app.layout = html.Div([

    html.H4('IT DataTable'),
    dt.DataTable(
        rows=df.to_dict('records'),
        #rows=df.iloc[1:,:].to_dict('records'),

        # optional - sets the order of columns
        #columns=sorted(df_dup.columns),

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
    return selected_row_indices + [0]


@app.callback(
    Output('graph-gapminder', 'figure'),
    [Input('datatable-gapminder', 'rows'),
     Input('datatable-gapminder', 'selected_row_indices')])
    
def update_figure(rows, selected_row_indices):
    #dff = pd.concat([df.iloc[0:1,:], pd.DataFrame(rows, columns=df.columns)])
    dff = pd.DataFrame(rows, columns=df.columns)
    
    fig = plotly.tools.make_subplots(
        rows=4, cols=1,
        subplot_titles=('UA_T', 'TA_T', 'DI_T', 'FC_T'),
        shared_xaxes=True)
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
