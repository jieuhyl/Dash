# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 16:40:48 2018

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
from os import path
from wordcloud import WordCloud
import base64

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
#from sklearn.preprocessing import StandardScaler
#from keras.models import model_from_json

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
#y_pred = mod_reg.predict(X_test)

# wc
d = "/Users/Jie.Hu/Desktop/social_dash"

# Read the whole text.
text = open(path.join(d, 'plot.txt')).read()

# Generate a word cloud image

wordcloud = WordCloud(colormap='hsv', background_color = 'black',
                      width = 800, height = 600, 
                      max_font_size= 100, min_font_size=10).generate(text)
#plt.figure(figsize = (8, 8), facecolor = None)
#plt.imshow(wordcloud, interpolation="bilinear")
#plt.axis("off")
#plt.show()

# store to file
wordcloud.to_file(path.join(d, "wc.png"))


image_filename = 'wc.png' # replace with your own image
encoded_image = base64.b64encode(open(image_filename, 'rb').read())

# prediction

available_indicators = np.array(['OBO', 'UA_T', 'TA_T', 'DI_T', 'FC_T'])


markdown_text ='''
**Method:** Artificial Neural Network. This may not be the best method here since data is too 'ez' for the network
which means the sample size is really samll and only three attributes are considered. The MAPE is around 25%.  
**Graph:** Implement 4D data to 2D graph. X-axis is the value for the reserved seat, Y-axis is actual value for OBO, bubble size is
proportional to theater count, bubble color reflects the rt score, you can you RHS colorscale bar as a reference.  
**Correlation:** The correlations of reserved seat at Wednesday afternoon, Wednesday night and
Thursday afternoon to OBO are 88%, 88% and 93% respectively. We are using Wednesday afternoon data.  
**Keep in Mind:** This is not an official forecast report but more like a tool to play with!
'''

# APP =========================================================================
app = dash.Dash()

colors = {
    'background': '#111111',
    'text': '#808080'
}

tab_style = {
    #'borderBottom': '1px solid #d6d6d6',
    #'padding': '10px',
    'fontWeight': 'bold'
}

tab_selected_style = {
    #'borderTop': '1px solid #d6d6d6',
    #'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'color': 'white'
    #'padding': '6px'
}

app.layout = html.Div([
    html.H1(
        children='Social Dash',
        style={
            'textAlign': 'center',
            'color': colors['text'],
            'margin': '5px',
            'font-family': 'Century Gothic'
        }
    ),

    html.H3(children='NRG, Data Science', 
        style={
        'textAlign': 'center',
        'color': colors['text'],
        'margin': '5px',
        'font-family': 'Century Gothic'
    }),
    
    html.H3(children='{} Summary'.format(df.iloc[0,0]),
            style={
            'textAlign': 'center',
            'color': colors['text'],
            'margin': '5px',
            'font-family': 'Century Gothic'
        }),
        
    dcc.Tabs(id="tabs", vertical=False,  children=[
        dcc.Tab(label='Tab 1', selected_style=tab_selected_style, children=[
        html.Div([
        html.H4('DataTable'),
        dt.DataTable(
        rows=df.to_dict('records'),

        # optional - sets the order of columns
        #columns=sorted(df_dup.columns),

        row_selectable=True,
        filterable=True,
        sortable=True,
        selected_row_indices=[],
        id='datatable'),
        html.Div(id='selected-indexes'),
        dcc.Graph(
        id='graph1'
       ),
            ], style={
        'textAlign': 'center',
        'margin': '5px',
        'color': colors['text'],
        'font-family': 'Century Gothic'
    })
        ], style={'width': '10%',
         'textAlign': 'center',
         'margin': '1px',
        'color': colors['text'],
        'font-family': 'Century Gothic',
        'fontWeight': 'bold'}),
        
    dcc.Tab(label='Tab 2', selected_style=tab_selected_style, children=[
      html.Div([
      html.H4('WordCloud'),
      html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()))
        ], style={
        'textAlign': 'center',
        'color': colors['text'],
        'margin': '5px',
        'font-family': 'Century Gothic'
    })], style={'width': '10%',
        'textAlign': 'center',
        'margin': '1px',
        'color': colors['text'],
        'font-family': 'Century Gothic',
        'fontWeight': 'bold'}),
                
   dcc.Tab(label='Tab 3', selected_style=tab_selected_style, children=[

    html.Div([dcc.Markdown(children = markdown_text)],
              style={
                     'color': colors['text'],
                     'font-family': 'Century Gothic'}),

   html.Div([
           html.Label('Theater Count: '),
           dcc.Slider(id='tc',
                   min=500,
                   max=4500,
                   step=100,
                   marks={i: '{}'.format(i) for i in range(500, 4600, 1000)},
                   value=3000)],
           style={'padding': '15px 15px 15px 0px',
                  'width': '65%',  'display': 'inline-block'}),

    html.Div([
           html.Label('Rotten Tomato: '),
           dcc.Slider(id='rt',
                   min=10,
                   max=100,
                   marks={i: '{}'.format(i) for i in range(10, 101,5)},
                   value=50)],
          style={'padding': '15px 15px 15px 0px',
                 'width': '65%', 'display': 'inline-block'}),
          
            
    html.Div(id='result',
                  style={
                 #'borderBottom': 'thin lightgrey solid',
                 #'backgroundColor': 'rgb(234,234,234)',
                 'padding': '15px 15px 15px 0px',
                 'font-weight': 'bold',
                 #'width': '100%',
                 #'margin': { 'l': 10, 'b': 20, 't': 30, 'r': 40},
                 #'font-family': 'Century Gothic'
                 }),

     html.Div([      
         html.Div([
                 html.Label('X-axis'), 
            dcc.Dropdown(
                id='crossfilter-xaxis-column',
                options=[{'label': i, 'value': i} for i in available_indicators],
                value=available_indicators[0]
            )],
         style={'width': '50%', 'display': 'inline-block'}),          
       
         html.Div([  
           html.Label('Y-axis'),    
           dcc.Dropdown(
                id='crossfilter-yaxis-column',
                options=[{'label': i, 'value': i} for i in available_indicators],
                value=available_indicators[0]
            )],
                style={'width': '50%', 'display': 'inline-block'})],
                   
               style={
                 #'borderBottom': 'thin lightgrey solid',
                 'backgroundColor': 'rgb(255,255,255)',
                 'padding': '10px 0px',
                 #'width': '100%',
                 #'margin': '20px, 10px, 30px, 40px',
                 'color': colors['text'],
                 'font-family': 'Century Gothic'
                 }),
    
    dcc.Graph(
        id='graph2',
        style={'height': 600}
    )
        ], style={'width': '10%',
                'textAlign': 'center',
             'margin': '1px',
             'color': colors['text'],
        'font-family': 'Century Gothic',
        'fontWeight': 'bold'}),
    ])
])

#1
@app.callback(
    Output('datatable', 'selected_row_indices'),
    [Input('graph1', 'clickData')],
    [State('datatable', 'selected_row_indices')])
def update_selected_row_indices(clickData, selected_row_indices):
    if clickData:
        for point in clickData['points']:
            if point['pointNumber'] in selected_row_indices:
                selected_row_indices.remove(point['pointNumber'])
            else:
                selected_row_indices.append(point['pointNumber'])
    return selected_row_indices +[0]

#2
@app.callback(
    Output('graph1', 'figure'),
    [Input('datatable', 'rows'),
     Input('datatable', 'selected_row_indices')])
    
def update_figure(rows, selected_row_indices):
    
    dff = pd.DataFrame(rows)
    
    fig = plotly.tools.make_subplots(
        rows=4, cols=1,
        subplot_titles=('UA_T', 'TA_T', 'DI_T', 'FC_T'),
        shared_xaxes=True, print_grid=False)
    marker = {'color': ['#0074D9']*len(dff)}
    for i in (selected_row_indices or []):
        marker['color'][i] = '#FF851B'
    fig.append_trace({
        'x': dff['MOVIE'],
        'y': dff['UA_T'],
        'type': 'bar',
        'marker': marker,
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
    fig['layout']['hovermode'] = 'compare'
    fig['layout']['height'] = 800
    fig['layout']['margin'] = {
        'l': 40,
        'r': 10,
        't': 60,
        'b': 200
        }
    return fig
# hover
# add reference line

#3
@app.callback(
     Output(component_id='result', component_property='children'),
    [Input(component_id='tc', component_property='value'),
     Input(component_id='rt', component_property='value')])
def update_three_inputs(tc, rt):
    if tc is not None and tc is not '' and rt is not None and rt is not '':
        try:
            Prediction = mod_reg.predict(np.array([[df_test.iloc[0,2], df_test.iloc[0,3], float(tc), float(rt)]]))[0]
            return 'With {} theater count and {} rotten tomato, the prediction is ${:,.1f}M. And confidence interval is from ${:,.1f}M to ${:,.1f}M.'.\
                format(tc, rt, Prediction/1000000, 0.77*Prediction/1000000, 1.23*Prediction/1000000, 2)
        
        except ValueError:
            return 'Unable to give the prediction'


#4  
@app.callback(
    dash.dependencies.Output('graph2', 'figure'),
    [Input('tc', 'value'),
     Input('rt', 'value'),
     dash.dependencies.Input('crossfilter-xaxis-column', 'value'),
     dash.dependencies.Input('crossfilter-yaxis-column', 'value')])
def update_graph(tc, rt, xaxis_column_name, yaxis_column_name):
    if tc is not None and tc is not '' and rt is not None and rt is not '':
     
        Prediction = mod_reg.predict(np.array([[df_test.iloc[0,2], df_test.iloc[0,3], float(tc), float(rt)]]))[0]
        df.iloc[0,1] = Prediction
        
    filtered_df = df.copy()
    
    return {
        'data': [go.Scatter(
            x=filtered_df[xaxis_column_name],
            y=filtered_df[yaxis_column_name],
            hoverinfo = 'text',
            mode='markers',
            marker={'size': filtered_df['DI_T'],
                    'color': filtered_df['FC_T'],
                    'colorscale': 'Picnic',
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
            hovermode='closest',
            annotations = [dict(x=filtered_df[xaxis_column_name][0], y=filtered_df[yaxis_column_name][0], text = 'New Movie')]
        )
    }
        
        
if __name__ == '__main__':
    app.run_server()