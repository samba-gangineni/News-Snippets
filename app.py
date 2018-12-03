from __future__ import print_function
from flask import Flask, render_template, url_for
import pickle
import random

__author__="Sambasiva Rao Gangineni"

#Creating the Flask application
app=Flask(__name__)

@app.route('/',methods=['GET','POST'])
@app.route('/home',methods=['GET','POST'])
def home():
    return render_template('index.html')

@app.route('/news',methods=['GET','POST'])
def news():
    title,dop,content,url = picking_news_for_display()
    print(content[0])
    return render_template('news.html',data=9,title=title,dop=dop,content = content,url=url)

@app.route('/about',methods=['GET','POST'])
def about():
    return render_template('about.html')

def picking_news_for_display():
    # Reading in the sorted news
    urls = pickle.load(open('url_sort.pkl','r'))
    titles = pickle.load(open('title_sort.pkl','r'))
    summaries = pickle.load(open('summary_sort.pkl','r'))
    dateofpublising = pickle.load(open('dop_sort.pkl','r'))

    to_display_indexes = []
    
    # Collecting the indexes for the display
    for i in range(9):
        to_display_indexes.append(random.sample(range(0,len(urls[i])),15))

    # Collecting the dop, title, summary and url to display
    dop_display = []
    url_display=[]
    title_display=[]
    summary_display=[]
    for i in range(len(to_display_indexes)):
        url_append=[]
        title_append = []
        dop_append = []
        summary_append = []
        for j in to_display_indexes[i]:
            url_append.append(urls[i][j])
            title_append.append(titles[i][j])
            dop_append.append(dateofpublising[i][j])
            summary_append.append(summaries[i][j])
        
        dop_display.append(dop_append)
        url_display.append(url_append)
        title_display.append(title_append)
        summary_display.append(summary_append)

    return title_display, dop_display, summary_display, url_display

if __name__=="__main__":
    picking_news_for_display()
    app.run(debug=True)