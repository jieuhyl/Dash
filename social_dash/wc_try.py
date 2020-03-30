# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 17:16:19 2018

@author: Jie.Hu
"""

# importing all necessery modules 
from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 
import pandas as pd 

d = "/Users/Jie.Hu/Desktop/social_dash"

# Read the whole text.
text = open(path.join(d, 'plot.txt')).read()

wordcloud = WordCloud(width = 800, height = 600,
                      colormap = 'hsv', background_color='black',
                      max_font_size=200, random_state=1234).generate(text)
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

    
# store to file
wordcloud.to_file(path.join(d, "wc.png"))
#==============================================================================
# frequency
df_wc = pd.read_csv('wc.csv')

mydict = dict(zip(df_wc.iloc[:,0], df_wc.iloc[:,1]))


wordcloud = WordCloud(width = 800, height = 600,
                      colormap = 'hsv', background_color='black',
                      max_font_size=400, min_font_size=10,
                      random_state=1234).generate_from_frequencies(frequencies=mydict)
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


#===================
# Reads 'Youtube04-Eminem.csv' file  
df = pd.read_csv(r"Youtube04-Eminem.csv", encoding ="latin-1") 
  
comment_words = ' '
stopwords = set(STOPWORDS) 
  
# iterate through the csv file 
for val in df.CONTENT: 
      
    # typecaste each val to string 
    val = str(val) 
  
    # split the value 
    tokens = val.split() 
      
    # Converts each token into lowercase 
    for i in range(len(tokens)): 
        tokens[i] = tokens[i].lower() 
          
    for words in tokens: 
    comment_words = comment_words + words + ' '
  
  
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(comment_words) 
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 