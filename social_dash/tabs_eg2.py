#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Created on Tue Oct  9 11:12:18 2018

@author: Jie.Hu
"""

import dash
import dash_html_components as html
import dash_core_components as dcc

from dash.dependencies import Input, Output

app = dash.Dash()

tabs_styles = {'height': '44px'}
tab_style = {'borderBottom': '1px solid #d6d6d6', 'padding': '10px',
             'fontWeight': 'bold'}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'color': 'white',
    'padding': '6px',
    }

app.layout = html.Div([dcc.Tabs(id='tabs-styled-with-inline',
                      value='tab-1', vertical=False,
                      children=[dcc.Tab(label='Tab 1', value='tab-1',
                      selected_style=tab_selected_style),
                      dcc.Tab(label='Tab 2', value='tab-2',
                      style=tab_style,
                      selected_style=tab_selected_style),
                      dcc.Tab(label='Tab 3', value='tab-3',
                      style=tab_style,
                      selected_style=tab_selected_style),
                      dcc.Tab(label='Tab 4', value='tab-4',
                      style=tab_style,
                      selected_style=tab_selected_style)],
                      style=tabs_styles),
                      html.Div(id='tabs-content-inline')])


@app.callback(Output('tabs-content-inline', 'children'),
              [Input('tabs-styled-with-inline', 'value')])
def render_content(tab):
    if tab == 'tab-1':
        return html.Div([html.H3('Tab content 1')])
    elif tab == 'tab-2':
        return html.Div([html.H3('Tab content 2')])
    elif tab == 'tab-3':
        return html.Div([html.H3('Tab content 3')])
    elif tab == 'tab-4':
        return html.Div([html.H3('Tab content 4')])

if __name__ == '__main__':
    app.run_server()

			