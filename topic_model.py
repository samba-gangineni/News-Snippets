'''
    Get the data from the google news api for testing the top modelling
'''
from __future__ import print_function
from nltk.tokenize import RegexpTokenizer
from stopwords import get_stopwords
from datetime import datetime
from gensim import models,corpora
from copy import deepcopy
import json
import sys
import pyLDAvis
import pyLDAvis.gensim
import operator
import pickle

'''
    Getting number of optimal topics based on the topic coherence
'''
def optimum_topics(corpus,list_topics,dictionary,iterations,processed_content):
    temp = -10000
    temp_model = None
    topics = 0
    model_coherences = []
    for i in range(0,len(list_topics)):
        lda_model = models.LdaModel(corpus=deepcopy(corpus),num_topics=list_topics[i],id2word=deepcopy(dictionary),iterations=iterations)

        #Calculating the coherence
        topic_coherence = models.CoherenceModel(model=lda_model, texts=processed_content, dictionary=dictionary, coherence='c_v')
        model_coherence = topic_coherence.get_coherence()
        model_coherences.append(model_coherence)
        if model_coherence>=temp:
            topics = list_topics[i]
            temp = model_coherence
            temp_model = lda_model
    
    return temp_model,topics,temp,model_coherences

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

#Building the LDA model with optimum topics
topics_list = range(2,12)
lda_model, num_topics, model_coherence, list_coherences = optimum_topics(corpus,topics_list,dictionary,150,processed_content)

'''
    Model Evaluation
'''
#model perplexity calculation
model_preplexity = lda_model.log_perplexity(corpus)

'''
    Visualising the model
'''
visualisation = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)

# Saving the visualisation to the html
pyLDAvis.save_html(visualisation,'topic_model.html')

# Labelling the document using the model
labels = []
for i in content:
    tokens = tokenizer.tokenize(i.lower())
    while 'reuters' in tokens or 's' in tokens or 'said' in tokens or 't' in tokens:
        if 'reuters' in tokens:
            tokens.remove('reuters')
        if 's' in tokens:
            tokens.remove('s')
        if 'said' in tokens:
            tokens.remove('said')
        if 't' in tokens:
            tokens.remove('t')
    
    stopped_tokens = [j for j in tokens if j not in english_stopwords]
    bow = dictionary.doc2bow(stopped_tokens)
    prediction = lda_model[bow]
    best_label = max(dict(prediction).iteritems(), key = operator.itemgetter(1))[0]
    labels.append(best_label)

'''
    Saving the important list for summarisation
'''
# Urls
with open('url.pkl','w') as f:
    pickle.dump(url,f)

# title
with open('title.pkl','w') as f:
    pickle.dump(title,f)

# dop
with open('dop.pkl','w') as f:
    pickle.dump(dop,f)

# content
with open('content.pkl','w') as f:
    pickle.dump(content,f)

# labels
with open('labels.pkl','w') as f:
    pickle.dump(labels,f)

#Saving the coherence list for plotting later
with open('coherence.pkl','w') as f:
    pickle.dump(list_coherences,f)

# saving the topics
lda_model.save('lda.model')