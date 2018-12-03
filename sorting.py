from __future__ import print_function
import pickle
import numpy as np

__author__="Sambasiva Rao Gangineni"


# Creating the list of cateogries for the news articles
url = []
dop = []
summary=[]
title = []

# Reading in the summarized_content, url, dop, labels, titles
labels = pickle.load(open('labels.pkl','r'))
urls = pickle.load(open('url.pkl','r'))
titles = pickle.load(open('title.pkl','r'))
summaries = pickle.load(open('summary.pkl','r'))
dateofpublising = pickle.load(open('dop.pkl','r'))

# as we have only 9 cateogories
label_np = np.array(labels)
url_np = np.array(urls)
dop_np = np.array(dateofpublising)
summaries_np = np.array(summaries)
titles_np = np.array(titles)

for i in range(9):
    bool_vector = label_np==i
    url.append(url_np[bool_vector].tolist())
    dop.append(dop_np[bool_vector].tolist())
    summary.append(summaries_np[bool_vector].tolist())
    title.append(titles_np[bool_vector].tolist())

# Saving them to new files
pickle.dump(url,open('url_sort.pkl','w'))
pickle.dump(dop,open('dop_sort.pkl','w'))
pickle.dump(summary,open('summary_sort.pkl','w'))
pickle.dump(title,open('title_sort.pkl','w'))