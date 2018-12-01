'''
    Get the data from the google news api for testing the top modelling
'''
from __future__ import print_function
import json


__author__="Sambasiva Rao Gangineni"

#Reading in the newsarticles
news_articles = []
with open('news-reuters-2014.json','r') as news:
    for article in news:
        news_articles.append(article)


print(len(news_articles))
