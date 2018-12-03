from __future__ import print_function
from __future__ import unicode_literals
from gensim import models,corpora
from nltk.tokenize import RegexpTokenizer
from stopwords import get_stopwords
import pickle
import operator
import pyLDAvis
import pyLDAvis.gensim
import sys
import spacy

__author__="Sambasiva Rao Gangineni"

'''
    Load all other lists url, title, text, dop etc.,
'''
# Urls
with open('url.pkl','r') as f:
    url = pickle.load(f)

# title
with open('title.pkl','r') as f:
    title = pickle.load(f)

# dop
with open('dop.pkl','r') as f:
    dop = pickle.load(f)

# content
with open('content.pkl','r') as f:
    content = pickle.load(f)

# Loading the preplexity list for plotting later
with open('preplexity.pkl','r') as f:
    prelexity_list = pickle.load(f)

# Loading the coherence list for plotting later
with open('coherence.pkl','r') as f:
    coherence_list = pickle.load(f)

# Loading the dictionary
with open('dictionary.pkl','r') as f:
    dictionary = pickle.load(f)

# Loading the corpus
with open('corpus.pkl','r') as f:
    corpus = pickle.load(f)

'''
    From the graph select the optimal number of topics and pick that particular model
'''
# Loading the topics
a = range(2,12)
topics = int(sys.argv[1])
i = a.index(topics)
lda_model = models.LdaModel.load('lda{}.model'.format(i))

'''
    Visualising the model
'''
visualisation = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)

# Saving the visualisation to the html
pyLDAvis.save_html(visualisation,'templates/topic_model.html')

'''
    Labelling the documents using the models
'''
tokenizer = RegexpTokenizer(r'[a-zA-Z]{3,}')
english_stopwords = get_stopwords('en')
english_stopwords.append('reuters')
english_stopwords.append('said')
token_content = []
processed_content = []
for i in content:
    tokens = tokenizer.tokenize(i.lower())
    token_content.append(tokens)
    stopped_tokens = [j for j in tokens if j not in english_stopwords]
    processed_content.append(stopped_tokens)
    
# Creating a bigram model
bigram = models.Phrases(token_content, min_count=5, threshold = 100)
bigram_mod = models.phrases.Phraser(bigram)
bigram_content = [bigram_mod[i] for i in processed_content]

# lemmatisation
nlp=spacy.load('en',disable=['parser','ner'])
allowed_postags = ['NOUN','ADJ','VERB','ADV']
lemmatised_content=[]
for each_article in bigram_content:
    try:
        doc = nlp(" ".join(each_article))
        lemmatised_content.append([tokens1.lemma_ for tokens1 in doc if tokens1.pos_ in allowed_postags])
    except:
        print(each_article)

# Creating the corpus
corpus_bigram = [dictionary.doc2bow(text) for text in lemmatised_content]

# Labelling
labels = []
for each_doc in corpus_bigram:
    prediction = lda_model[each_doc]
    best_label = max(dict(prediction).iteritems(), key = operator.itemgetter(1))[0]
    labels.append(best_label)

'''
    Saving the labels for display
'''
 # Saving the labels
with open('labels.pkl','w') as f:
    pickle.dump(labels,f)