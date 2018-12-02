'''
    Get the data from the google news api for testing the top modelling
'''
from __future__ import print_function
from nltk.tokenize import RegexpTokenizer
from stopwords import get_stopwords
from datetime import datetime
from gensim import models,corpora
import json
import sys
import re

__author__="Sambasiva Rao Gangineni"



'''
    Preprocessing the data
'''
starttime1 = datetime.now()
#Reading in the 50000 newsarticles
url = []
title = []
dop = []
content = []
count =  0
ten_thousand = 0
with open(sys.argv[1],'r') as news:
    for article in news:
        if ten_thousand<=10000:
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
                ten_thousand+=1

            except:
                count+=1

starttime2 = datetime.now()
print("Total time for reading",starttime2-starttime1)

#Tokenising, stopping and stemming the content
tokenizer = RegexpTokenizer(r'[a-zA-Z]{3,}')
english_stopwords = get_stopwords('en')
english_stopwords.append('reutuers')

processed_content = []
for article in content:
    tokens = tokenizer.tokenize(article.lower())
    while 'reuters' in tokens or 's' in tokens or 'said' in tokens or 't' in tokens:
        if 'reuters' in tokens:
            tokens.remove('reuters')
        if 's' in tokens:
            tokens.remove('s')
        if 'said' in tokens:
            tokens.remove('said')
        if 't' in tokens:
            tokens.remove('t')
    
    stopped_tokens = [i for i in tokens if i not in english_stopwords]
    processed_content.append(stopped_tokens)

'''
    Topic modelling using LDA
'''
# Building a dictionary of the all documents
dictionary = corpora.Dictionary(processed_content)

# Building the bag of words representation for each document
corpus = [dictionary.doc2bow(text) for text in processed_content]

#Building the LDA model
lda_model = models.LdaModel(corpus=corpus,num_topics=10,id2word=dictionary,iterations=500)

print("LDA Model:")
for i in range(10):
    print("topic {}:{}".format(i,lda_model.print_topic(i,10)))

'''
    Model Evaluation
'''
#model perplexity calculation
model_preplexity = lda_model.log_perplexity(corpus)

# Topic coherence calculation
topic_coherence = models.CoherenceModel(model=lda_model, texts=processed_content, dictionary=dictionary, coherence='c_v')
model_coherence = topic_coherence.get_coherence()
print("Perplexity {}".format(model_preplexity))
print("Coherence {}".format(model_coherence))

'''
    Visualising the model
'''
starttime3 = datetime.now()
print("Total time for reading",starttime3-starttime2)