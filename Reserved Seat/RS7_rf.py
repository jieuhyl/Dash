# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 14:06:46 2018

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
#import matplotlib.pyplot as plt
#import seaborn as sns
#sns.set_style("whitegrid")
#from scipy import stats
#from scipy.stats import norm, skew 

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor


# read data
df_new = pd.read_csv('df_new.csv')

# check missing
#df_new.isnull().values.sum()
#missing_ratio = df_new.isnull().sum() / len(df_new)


''' random forest '''
# get mtx
y = np.log(df_new['OBO']).values
df_new['RS_W_A_log'] = np.log(df_new['RS_W_A'])
X = df_new.iloc[:, [10,5,6]].values

# Feature Scaling StandardScaler(), MinMaxScaler()
sc = MinMaxScaler()
X = sc.fit_transform(X)

reg = RandomForestRegressor(bootstrap = True,
                               max_depth = 6, 
                               max_features = 'sqrt', 
                               min_samples_leaf = 1, 
                               min_samples_split = 3,
                               n_estimators=100,
                               random_state= 1337)
reg.fit(X,y)

# Predicting the Test set results
y_pred = reg.predict(X)
# check r2 and mape
np.mean(np.abs((np.exp(y) - np.exp(y_pred)) / np.exp(y))) * 100

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
**Method:** Artificial Neural Network, this is not a linear model like theatrical model or social model.
So if you scale one attribute up while holding the rest, the prediction may not be necessary go up.  
**Graph:** Implement 4D data to 2D graph. X-axis is the value for the reserved seat, Y-axis is actual value for OBO, bubble size is
proportional to theater count, bubble color reflects the rt score, you can you RHS colorscale bar as a reference.  
**Correlation:** The correlations of reserved seat at Wednesday afternoon, Wednesday night and
Thursday afternoon to OBO are 88%, 88% and 93% respectively. We are using Wednesday afternoon data.  
**Keep in Mind:** I have deleted a few blockbusters: Avengers - Infinity War, Star Was - The Last Jedi and Black Panther which are examined as outliers. 
Last, This is not an official forecast report but more like a tool to play with! 
'''

'''
### Dash apps can be written in Markdown.
Dash uses the [CommonMark](http://commonmark.org/)
specification of Markdown.\
Check out their [60 Second Markdown Tutorial](http://commonmark.org/help/)l
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
             html.Label('Reserved Seat: '),    
              dcc.Input(id='seat', placeholder='Enter a value...', type='text')],
              style={'width': '20%', 'display': 'inline-block'}),           
       
        html.Div([  
           html.Label('Theater Count: '),    
           dcc.Input(id='screen', placeholder='Enter a value...', type='text')],
           style={'width': '20%', 'display': 'inline-block'}),
                   
        html.Div([  
           html.Label('Rotten Tomato: '),            
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
            Prediction = np.exp(reg.predict(sc.transform([[np.log(float(seat)), float(screen), float(rt)]]))).tolist()[0]
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
    Prediction = np.exp(reg.predict(sc.transform([[np.log(float(selected_seat)), float(selected_screen), float(selected_rt)]])).tolist())[0]
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