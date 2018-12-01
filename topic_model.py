'''
    Get the data from the google news api for testing the top modelling
'''
from __future__ import print_function
from nltk.tokenize import RegexpTokenizer
import json


__author__="Sambasiva Rao Gangineni"

'''
    Preprocessing the data
'''
#Reading in the newsarticles
news_content = []
url = []
title=[]
dop=[]
count = 0
with open('news-reuters-2014.json','r') as news:
    for article in news:
        try:
            test = json.loads(article)
            urls = test.get('url')
            dops = test.get('dop')
            texts = test.get('text')
            titles = test.get('title')
            if '(Reuters)' in texts and '(Reporting by ' in texts:
                texts = texts[texts.index('(Reuters)')+9:texts.index('(Reporting by ')]
            news_content.append(texts)
            url.append(urls)
            title.append(titles)
            dop.append(dops)
        except:
            count+=1

# Tokenising the words


