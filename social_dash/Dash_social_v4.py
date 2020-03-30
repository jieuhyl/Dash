# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 14:14:45 2018

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

# APP
app = dash.Dash()

available_indicators = np.array(['OBO', 'UA_T', 'TA_T', 'DI_T', 'FC_T'])

app.layout = html.Div([
    html.Div([      
         html.Div([
                 html.Label('X'), 
            dcc.Dropdown(
                id='crossfilter-xaxis-column',
                options=[{'label': i, 'value': i} for i in available_indicators],
                value=available_indicators[0]
            )],
         style={'width': '33%', 'display': 'inline-block'}),          
       
         html.Div([  
           html.Label('Y'),    
           dcc.Dropdown(
                id='crossfilter-yaxis-column',
                options=[{'label': i, 'value': i} for i in available_indicators],
                value=available_indicators[0]
            )],
                style={'width': '33%', 'display': 'inline-block'}),
                   
         html.Div([  
           html.Label('Similarity'),            
           dcc.RangeSlider(
               id='similarity-slider',
               min=0,
               max=1,
               value=[df['Similarity'].min(), df['Similarity'].max()],
               step=0.1,
               marks={
                      0: {'label': 0, 'style': {'color': '#77b0b1'}},
                      0.5: {'label': 0.5},
                      0.6: {'label': 0.6},
                      0.7: {'label': 0.7},
                      0.8: {'label': 0.8},
                      0.9: {'label': 0.9},
                      1.0: {'label': 1, 'style': {'color': '#f50'}}
                    }
           )],
               style={'width': '33%', 'float': 'right', 'display': 'inline-block'})],
                   
               style={
                 #'borderBottom': 'thin lightgrey solid',
                 'backgroundColor': 'rgb(255,255,255)'
                 #'padding': '10px 5px'
                 #'width': '100%',
                 #'margin': '20px, 10px, 30px, 40px',
                 #'font-family': 'Century Gothic'
                 }),

    dcc.Graph(
        id='graph',
        style={'height': 600}
           )
          ],
                 style={'font-family': 'Century Gothic',
                        'margin': '5px'
        })




@app.callback(
    dash.dependencies.Output('graph', 'figure'),
    [dash.dependencies.Input('crossfilter-xaxis-column', 'value'),
     dash.dependencies.Input('crossfilter-yaxis-column', 'value'),
     dash.dependencies.Input('similarity-slider', 'value')])
def update_graph(xaxis_column_name, yaxis_column_name, selected_similarity):
     
    filtered_df = df[(df['Similarity'] >= selected_similarity[0]) & 
                     (df['Similarity'] <= selected_similarity[1])]
    
    return {
        'data': [go.Scatter(
            x=filtered_df[xaxis_column_name],
            y=filtered_df[yaxis_column_name],
            hoverinfo = 'text',
            mode='markers',
            marker={'size': filtered_df['Similarity']*20,
                    'color': filtered_df['Genre'],
                    'showscale': True,
                    'line': {'width': 0.5, 'color': 'white'}
                    }
        )],
    
        'layout': go.Layout(
            xaxis={
                'title': xaxis_column_name
                  },
            yaxis={
                'title': yaxis_column_name
                  },
            margin={'l': 40, 'b': 30, 't': 10, 'r': 0},
            height=450,
            hovermode='closest'
        )
    }


if __name__ == '__main__':
    app.run_server()
