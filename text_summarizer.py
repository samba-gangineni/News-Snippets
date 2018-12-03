from __future__ import print_function
from __future__ import unicode_literals
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np

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
def build_similarity_matrix():
    pass

