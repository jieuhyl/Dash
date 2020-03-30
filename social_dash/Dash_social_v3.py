# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 13:21:01 2018

@author: Jie.Hu
"""
import pandas as pd
import dash
import dash_html_components as html
import base64

from os import path
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# read wc 
d = "/Users/Jie.Hu/Desktop/social_dash"

df_wc = pd.read_csv('wc.csv')

mydict = dict(zip(df_wc.iloc[:,0], df_wc.iloc[:,1]))


wordcloud = WordCloud(width = 800, height = 600,
                      colormap = 'hsv', background_color='white',
                      max_font_size=400, min_font_size=10,
                      random_state=1234).generate_from_frequencies(frequencies=mydict)
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# store to file
wordcloud.to_file(path.join(d, "wc.png"))

app = dash.Dash()

colors = {
    'background': '#111111',
    'text': '#808080'
}

image_filename = 'wc.png' # replace with your own image
encoded_image = base64.b64encode(open(image_filename, 'rb').read())
app.layout = html.Div([
        html.H1(
        children='International Tracking Correlation',
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
        
    html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()))
])

if __name__ == '__main__':
    app.run_server()