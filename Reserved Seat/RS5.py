# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 11:59:22 2018

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

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('RS_nrg.csv', encoding='cp1252')


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


df_new['RS_W_N'][df_new['NAME'] == 'Just Getting Started'] = 0
df_new['RS_T_A'][df_new['NAME'] == 'Just Getting Started'] = 0
df_new['RS_T_A'][df_new['NAME'] == 'Teen Titans Go! To the Movies'] = 1300
df_new['RS'][df_new['NAME'] == 'Teen Titans Go! To the Movies'] = 2580
df_new['RS_T_A'][df_new['NAME'] == 'Mission: Impossible - Fallout'] = 13000
df_new['RS'][df_new['NAME'] == 'Mission: Impossible - Fallout'] = 25357


#RT for less than 5 to 5 and RS less than 10 to 10
df_new.loc[df_new['RT']<=5, 'RT'] = 5
df_new.loc[df_new['RS_W_A']<=10, 'RS_W_A'] = 10


''' Modeling '''
# get mtx
y = df_new['OBO'].values
df_new['RS_W_A_log'] = np.log(df_new['RS_W_A'])
X = df_new.iloc[:, [10,5,6]].values

# Feature Scaling
sc = StandardScaler()
X = sc.fit_transform(X)
y = np.log(y)


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


markdown_text = '''
Method: Artificial Neural Network, this is not a linera model like theatrical model or social model.
So if you scale one attribute up while holding the rest, the prediction may not be necessary go up. \n
Correlation: The correlations of reserved seat at Wednesday afternoon, Wednesday night and
Thursday afternoon to OBO are 88%, 88%, 93% respectively. We are using Wednesday afternoon data. \n
Keep in mind: This is not an official forecast report but more like a tool to play with!
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

    html.Div(children='Marketing Science, NRG', style={
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
],
                 style={'font-family': 'Century Gothic'
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
         

if __name__ == '__main__':
    app.run_server()
