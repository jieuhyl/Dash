# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 01:47:05 2018

@author: Jie.Hu
"""

import dash
import dash_html_components as html
import dash_core_components as dcc

from dash.dependencies import Input, Output, State



app = dash.Dash()

app.layout = html.Div([
    dcc.Tabs(id="tabs", vertical=True, children=[
        dcc.Tab(label='Tab one', children=[
            html.Div([
                dcc.Graph(
                    id='example-graph',
                    figure={
                        'data': [
                            {'x': [1, 2, 3], 'y': [4, 1, 2],
                                'type': 'bar', 'name': 'SF'},
                            {'x': [1, 2, 3], 'y': [2, 4, 5],
                             'type': 'bar', 'name': u'Montréal'},
                        ]
                    }
                )
            ])
        ]),
        dcc.Tab(label='Tab two', children=[
                dcc.Graph(
                    id='example-graph-1',
                    figure={
                        'data': [
                            {'x': [1, 2, 3], 'y': [1, 4, 1],
                                'type': 'bar', 'name': 'SF'},
                            {'x': [1, 2, 3], 'y': [1, 2, 3],
                             'type': 'bar', 'name': u'Montréal'},
                        ]
                    }
                )
        ]),
        dcc.Tab(label='Tab three', children=[
                dcc.Graph(
                    id='example-graph-2',
                    figure={
                        'data': [
                            {'x': [1, 2, 3], 'y': [2, 4, 3],
                                'type': 'bar', 'name': 'SF'},
                            {'x': [1, 2, 3], 'y': [5, 4, 3],
                             'type': 'bar', 'name': u'Montréal'},
                        ]
                    }
                )
        ]),
    ])
])


if __name__ == '__main__':
    app.run_server()