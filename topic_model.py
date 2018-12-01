'''
    Get the data from the google news api for testing the top modelling
'''
from __future__ import print_function
from nltk.tokenize import RegexpTokenizer
from stopwords import get_stopwords
from nltk.stem.porter import PorterStemmer
from datetime import datetime
import json
import sys

__author__="Sambasiva Rao Gangineni"



'''
    Preprocessing the data
'''
#Reading in the 50000 newsarticles
url = []
title = []
dop = []
content = []
count =  0
fifty_thousand = 0
with open(sys.argv[1],'r') as news:
    for article in news:
        if fifty_thousand<=49999:
            try:
                # Loadings the json object
                test = json.loads(article)

                #Checking whether the article has all the components
                urls = test.get('url')
                titles = test.get('title')
                dops = test.get('dop')
                texts = test.get('text')

                #Removing the reporters name
                if '(Reuters)' in texts and '(Reporting by ' in texts:
                    texts = texts[texts.index('(Reuters)')+9:texts.index('(Reporting by ')]
                
                #Appending the articles
                url.append(urls)
                title.append(titles)
                dop.append(dops)
                content.append(texts)

                #Condition for reading in the 50000 articles
                fifty_thousand+=1
                
            except:
                count+=1

    
    

