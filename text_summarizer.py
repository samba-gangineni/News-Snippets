from __future__ import print_function
from __future__ import unicode_literals
from stopwords import get_stopwords
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
from operator import itemgetter
import numpy as np
import pickle

np.seterr(divide='ignore', invalid='ignore')

__author__="Sambasiva Rao"
''' 
    Summarisation using the Text Rank 
    - Text rank algorithm 
    - Calculating the similarity between the sentences
    - Building a similarity matrix between the sentences
    - Normalising the similarity along the row 
'''
# Text rank algorithm
def text_rank(similarity_matrix,precision,damping_factor):
    vector_length = len(similarity_matrix)
    
    # Initializing the rank vector
    rank_vector = np.ones(vector_length)/vector_length
    # Algorithm
    while True:
        new_ranks = (np.ones(vector_length)*(1-damping_factor)/vector_length)+damping_factor*(similarity_matrix.T.dot(rank_vector))
        change = abs(new_ranks-rank_vector).sum()
        if change<=precision:
            return new_ranks
        rank_vector = new_ranks

# Calculating the similarity between the sentences
def sentence_similarity(sent1,sent2,stopwords=None):
    
    # checking if the stop words are provided are not.
    if stopwords is None:
        stopwords=[]
    
    # lowering all the words in each sentence
    sent1 = [word.lower() for word in sent1]
    sent2 = [word.lower() for word in sent2]

    # Creating a list of unique words in both the sentences
    allwords = list(set(sent1+sent2))

    # building two vectors for calculating the similarities
    vector1 = [0]*len(allwords)
    vector2 = [0]*len(allwords)

    #vector for first sentence
    for word in sent1:
        if word in stopwords:
            continue
        vector1[allwords.index(word)]+=1
    
    # Vector for the second sentence
    for word in sent2:
        if word in stopwords:
            continue
        vector2[allwords.index(word)]+=1
    
    # Returning the similarity
    return 1-cosine_distance(vector1,vector2)

# constructing a sentence similarity matrix
def similarity_matrix(sentences, stopwords=None):
    
    # number of sentences
    num_sentences = len(sentences)

    #Intializing the matrix with zeros
    similarity_matrix = np.zeros((num_sentences,num_sentences))

    # Computing the similarity between two sentences and adding to the matrix
    for i in range(num_sentences):
        for j in range(num_sentences):
            # As we will be comparing the distance between the same sentence, hence we will skip it
            if i==j:
                continue
            similarity_matrix[i][j] = sentence_similarity(sentences[i],sentences[j],stopwords)

    # Normalising the matrix
    for i in range(num_sentences):
        if similarity_matrix[i].sum()!=0:
            similarity_matrix[i]=similarity_matrix[i]/similarity_matrix[i].sum()
    
    return similarity_matrix

# summarising and getting top sentences
def summarize_article(sentences,num_of_sentences, stopwords = None):

    # Calculating the similarity matrix for the article
    s_matrix = similarity_matrix(sentences, stopwords)

    # Generating the rank vector
    sentence_ranks = text_rank(s_matrix,0.0001,0.85)

    # Sorting the sentence ranks and taking the required number of sentences
    ranked_sentence_indexes = [item[0] for item in sorted(enumerate(sentence_ranks), key=lambda item: -item[1])]
    selected_sentences = sorted(ranked_sentence_indexes[:num_of_sentences])
    summary = itemgetter(*selected_sentences)(sentences)
    return summary

# Loading the news content
with open('content.pkl','r') as f:
    content = pickle.load(f)

# stop words
english_stopwords = get_stopwords('en')
english_stopwords.append('-')
english_stopwords.append('*')
english_stopwords.append('_')

# Summarising all the articles
summarized_content = []
for article in content:
    article = article.split('.')
    while '' in article or '*' in article or '-' in article:
        if '' in article:
            article.remove('')
        if '*' in article:
            article.remove('*')
        if '-' in article:
            article.remove('-')
    
    sentences = []
    for sentence in article:
        se.append(sentence.split())
    
    #summarising
    result = summarize_article(sentences,1,english_stopwords)

    # making the result into a sentence
    summary = " ".join(result[0])
    print(summary)
    # Appending the result to a list
    summarized_content.append(summary)

'''
    Saving the summarised content
'''
with open('summary.pkl','w') as f:
    pickle.dump(summarized_content,f)