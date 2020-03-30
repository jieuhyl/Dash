# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 13:35:52 2018

@author: Jie.Hu
"""

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
import dash_auth
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
from os import path
from wordcloud import WordCloud
import base64

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import cosine_similarity
#from sklearn.preprocessing import StandardScaler
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
#y_pred = mod_reg.predict(X_test)

# wc
d = "/Users/Jie.Hu/Desktop/social_dash"

# Read the whole text.
df_wc = pd.read_csv('wc.csv')

mydict = dict(zip(df_wc.iloc[:,0], df_wc.iloc[:,1]))


wordcloud = WordCloud(width = 800, height = 600,
                      colormap = 'hsv', background_color='black',
                      max_font_size=400, min_font_size=10,
                      random_state=1234).generate_from_frequencies(frequencies=mydict)
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
user_password = [['username', 'password'], ['nrgds', '1234']]

app = dash.Dash()

auth = dash_auth.BasicAuth(app, user_password)

server = app.server

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

# dropdown_1
studio_options = df['Studio'].unique()
# dropdown_2
genre_options = df['Genre'].unique()

hover_text = []

for index, row in df.iterrows():
    hover_text.append(('Movie Name: {mvname}<br>'+
                       'Year: {yr:,.0f}<br>'+
                       'Unaided Awareness: {ua:,.0f}<br>'+
                       'Total Awareness: {ta:,.0f}<br>'+
                       'Genre: {genre}<br>'+
                       'OBO: ${value:,.0f}').format(mvname=row['MOVIE'],
                                                    yr=row['Year'],
                                                    ua=row['UA_T'],
                                                    ta=row['TA_T'],
                                                    genre=row['Genre'],
                                                    value=row['OBO']))

df['Text'] = hover_text

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
        
    dcc.Tabs(id="tabs", vertical=False, style={'textAlign': 'center'}, children=[
        dcc.Tab(label='Tab 1', selected_style=tab_selected_style, children=[
        html.Div([
         html.Div([       
             html.Label('Studio'),    
            dcc.Dropdown(
                id='studio-picker',
                options=[{'label': i, 'value': i} for i in studio_options],
                value=[studio_options[0]],
                multi=True
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
        id='graph1'
       ),
            ], style={
        #'textAlign': 'center',
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
                   min=df['DI_T'].min(),
                   max=df['DI_T'].max(),
                   step=2,
                   marks={i: '{}'.format(i) for i in range(df['DI_T'].min(),df['DI_T'].max()+1,2)},
                   value=df['DI_T'].mean())],
           style={'padding': '15px 15px 15px 0px',
                  'width': '65%',  'display': 'inline-block',
                  'color': colors['text'],
                  'font-family': 'Century Gothic'}),

    html.Div([
           html.Label('Rotten Tomato: '),
           dcc.Slider(id='rt',
                   min=df['FC_T'].min(),
                   max=df['FC_T'].max(),
                   marks={i: '{}'.format(i) for i in range(df['FC_T'].min(),df['FC_T'].max()+1,2)},
                   value=df['FC_T'].mean())],
          style={'padding': '15px 15px 15px 0px',
                 'width': '65%', 'display': 'inline-block',
                 'color': colors['text'],
                 'font-family': 'Century Gothic'}),
          
            
    html.Div(id='result',
                  style={
                 #'borderBottom': 'thin lightgrey solid',
                 #'backgroundColor': 'rgb(234,234,234)',
                 'padding': '15px 15px 15px 0px',
                 'font-weight': 'bold',
                 #'width': '100%',
                 'color': colors['text'],
                 'font-family': 'Century Gothic'
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
]
    #style={'textAlign': 'center'}
    )


#0    
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
   filtered_df.loc[:,['UA_T', 'TA_T','DI_T', 'FC_T']] = filtered_df.loc[:,['UA_T', 'TA_T','DI_T', 'FC_T']].astype(float)  
   X_total = filtered_df.iloc[:,2:6].values
   mm = MinMaxScaler()
   X_total = mm.fit_transform(X_total)
   
   #gmm = GaussianMixture(n_components=4, covariance_type='full', random_state=1337).fit(X_total)
   n_components = np.arange(3, 6)
   models = [GaussianMixture(n, covariance_type='full', random_state=1337).fit(X_total)
             for n in n_components]
   gmm = GaussianMixture(n_components[[m.bic(X_total) for m in models].index(min([m.bic(X_total) for m in models]))], 
                                            covariance_type='full', random_state=1337).fit(X_total)
   
   filtered_df['Class'] = gmm.predict(X_total)

   filtered_df['Similarity_1'] = cosine_similarity(gmm.predict_proba(X_total)[0:1,:], 
                                        gmm.predict_proba(X_total))[0]

   filtered_df['Similarity_2'] = cosine_similarity(X_total[0:1,:], 
                                       X_total)[0]
   
   filtered_df.loc[:, 'Pct_1'] = filtered_df['Similarity_1'].rank(pct=True)
   filtered_df.loc[:, 'Pct_2'] = filtered_df['Similarity_2'].rank(pct=True)
   filtered_df.loc[:, 'Similarity'] = (filtered_df['Pct_1'] + filtered_df['Pct_2'])/2
   filtered_df['Similarity'] = filtered_df['Similarity'].round(2)  
   
   #filtered_df['Similarity'] = (filtered_df['Similarity_1']+filtered_df['Similarity_2'])/2
   filtered_df.drop(['Similarity_1', 'Similarity_2', 'Pct_1', 'Pct_2'], axis=1, inplace=True)  
   filtered_df.iloc[0,10] = 1
        
   return filtered_df.to_dict('records')
    
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
        df.iloc[0,4] = float(tc)
        df.iloc[0,5] = float(rt)
        
    filtered_df = df.copy()
    
    return {
        'data': [go.Scatter(
            x=filtered_df[xaxis_column_name],
            y=filtered_df[yaxis_column_name],
            text = df['Text'],
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
            annotations = [dict(x=filtered_df[xaxis_column_name][0], 
                                y=filtered_df[yaxis_column_name][0], 
                                text = '{}'.format(df.iloc[0,0]))]
        )
    }
        
# port=8181, host='10.16.21.110'        
if __name__ == '__main__':
    app.run_server()