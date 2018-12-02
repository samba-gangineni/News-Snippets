'''
    Get the data from the google news api for testing the top modelling
'''
from __future__ import print_function
from nltk.tokenize import RegexpTokenizer
from stopwords import get_stopwords
from datetime import datetime
from gensim import models,corpora
from copy import deepcopy
from pprint import pprint
import json
import sys
import matplotlib.pyplot as plt
import spacy
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
    model_preplexities = []
    models_tosave = []
    for i in range(0,len(list_topics)):
        lda_model = models.LdaModel(corpus=deepcopy(corpus),num_topics=list_topics[i],id2word=deepcopy(dictionary),iterations=iterations)
        models_tosave.append(lda_model)
        #model perplexity calculation
        model_preplexity = lda_model.log_perplexity(corpus)
        model_preplexities.append(model_preplexity)
        #Calculating the coherence
        topic_coherence = models.CoherenceModel(model=lda_model, texts=processed_content, dictionary=dictionary, coherence='c_v')
        model_coherence = topic_coherence.get_coherence()
        model_coherences.append(model_coherence)

        if model_coherence>=temp:
            topics = list_topics[i]
            temp = model_coherence
            temp_model = lda_model
    
    return temp_model,topics,temp,model_coherences, model_preplexities, models_tosave

__author__="Sambasiva Rao Gangineni"

'''
    Preprocessing the data
'''

#Reading in the 20000 newsarticles
url = []
title = []
dop = []
content = []
count =  0
twenty_thousand = 0
with open(sys.argv[1],'r') as news:
    for article in news:
        if twenty_thousand<=20000:
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
                twenty_thousand+=1

            except:
                count+=1

#Tokenising
tokenizer = RegexpTokenizer(r'[a-zA-Z]{3,}')
english_stopwords = get_stopwords('en')
english_stopwords.append('reutuers')
token_content = []
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
    token_content.append(tokens)
    stopped_tokens = [i for i in tokens if i not in english_stopwords]
    processed_content.append(stopped_tokens)


# Creating a bigram model
bigram = models.Phrases(token_content, min_count=5, threshold = 100)
bigram_mod = models.phrases.Phraser(bigram)
bigram_content = [bigram_mod[i] for i in processed_content]

# Lemmatising using spacy
nlp = spacy.load('en',disable=['parser','ner'])
allowed_tags = ['NOUN','ADJ','VERB','ADV']
lemmatised_content = []
for i in bigram_content:
    docu = nlp(" ".join(i))
    lemmatised_content.append([j.lemma_ for j in docu if j.pos_ in allowed_tags])

'''
    Topic modelling using LDA
'''
# Building a dictionary of the all documents
dictionary = corpora.Dictionary(processed_content)

# Building the bag of words representation for each document
corpus = [dictionary.doc2bow(text) for text in processed_content]

#Building the LDA model with optimum topics
topics_list = range(2,12)
lda_model, num_topics, model_coherence, list_coherences, list_preplexity, models_tosave = optimum_topics(corpus,topics_list,dictionary,350,processed_content)

'''
    Evaluating the model
'''

# Line graph for model_coherence to select the optimal number of topics
x = range(2,12,1)
plt.plot(x,list_coherences)
plt.xlabel("Num of topics")
plt.ylabel("Coherence score")
plt.legend(("Coherence values"),loc = 'best')
plt.savefig('coherence.png')
plt.clf()

# Line graph for model preplexity to select the optimal number of topics
plt.plot(x,list_preplexity)
plt.xlabel("Num of topics")
plt.ylabel("preplexity")
plt.legend(("preplexity values"),loc = 'best')
plt.savefig('preplexity.png')
plt.clf()

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

#Saving the preplexity list for plotting later
with open('preplexity.pkl','w') as f:
    pickle.dump(list_preplexity,f)

#Saving the coherence list for plotting later
with open('coherence.pkl','w') as f:
    pickle.dump(list_coherences,f)

# saving the topics
for i in range(0,len(models_tosave)):
    models_tosave[i].save('lda{}.model'.format(i))