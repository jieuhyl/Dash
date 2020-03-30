# -*- coding: utf-8 -*-
"""
Created on Thu Aug 09 16:42:24 2018

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
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score  
from sklearn.grid_search import GridSearchCV 

df = pd.read_csv('rs.csv')

# get mtx
y = df['OBO'].values
X = df.iloc[:, 2:].values

'''Linear Regession'''
reg = linear_model.LinearRegression()

# Train the model using the training sets
reg.fit(X, y)

# Make predictions using the testing set
y_pred = reg.predict(X)

#mean_squared_error(y, y_pred)
#r2_score(y, y_pred)



app = dash.Dash()



app.layout = html.Div(children=[
    html.H1(children='Linear Regression for Reserved Seat', style={'textAlign': 'center'}),

      html.Div([      
         html.Div([       
             html.Label('Seat'),    
              dcc.Input(id='seat', placeholder='Enter a value...', type='text')],
                style={'width': '33%', 'display': 'inline-block'}),           
       
         html.Div([  
           html.Label('Screen'),    
           dcc.Input(id='screen', placeholder='Enter a value...', type='text')],
                style={'width': '33%', 'display': 'inline-block'}),
                   
         html.Div([  
           html.Label('RT'),            
           dcc.Input(id='rt', placeholder='Enter a value...', type='text'),
            html.Div(id='result')],
               style={
                 'borderBottom': 'thin lightgrey solid',
                 'backgroundColor': 'rgb(255,255,255)',
                 'padding': '10px 5px',
                 #'width': '100%',
                 'margin': '5px',
                 'font-family': 'Century Gothic'
                 })]),



    dcc.Graph(
        id='scatter-plot',
        figure={
            'data': [
                go.Scatter(
                    x=df['SEAT'],
                    y=df['OBO'],
                    mode='markers',
                    opacity=0.7,
                    marker={
                        'size': df['SCRN'],
                        'color': df['RT'],
                        'colorscale': 'Viridis',
                        'showscale': True,
                        'line': {'width': 0.5, 'color': 'white'}
                    },
                )
            ],
            'layout': go.Layout(
                xaxis={'type': 'log', 'title': 'SEAT#'},
                yaxis={'title': 'OBO'},
                margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                hovermode='closest'
            )
        }
    )
])


@app.callback(
    Output(component_id='result', component_property='children'),
    [Input(component_id='seat', component_property='value'),
     Input(component_id='screen', component_property='value'),
     Input(component_id='rt', component_property='value')])
def update_years_of_experience_input(seat, screen, rt):
    if seat is not None and seat is not '' and screen is not None and screen is not '' and rt is not None and rt is not '':
        try:
            Prediction = reg.predict([float(seat), float(screen), float(rt)])[0]
            return 'With {} years of experience, {} screen count and {} RT, Prediction of ${:,.2f}'.\
                format(seat, screen, rt, Prediction, 2)
        except ValueError:
            return 'Unable to give the prediction'


if __name__ == '__main__':
    app.run_server()