# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 12:58:16 2018

@author: Jie.Hu
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 03:21:00 2017

@author: J
         Movie_World Cloud
"""
#import os

from os import path
from wordcloud import WordCloud
import matplotlib.pyplot as plt

d = "/Users/Jie.Hu/Desktop/social_dash"

# Read the whole text.
text = open(path.join(d, 'plot.txt')).read()

# Generate a word cloud image
wordcloud = WordCloud().generate(text)

# Display the generated image:
# the matplotlib way:
#plt.imshow(wordcloud, interpolation='bilinear')
#plt.axis("off")

# lower max_font_size
wordcloud = WordCloud(max_font_size=60).generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# store to file
wordcloud.to_file(path.join(d, "wc.png"))