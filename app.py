from __future__ import print_function
from flask import Flask, render_template

__author__="Sambasiva Rao Gangineni"

#Creating the Flask application
app=Flask(__name__)

@app.route('/',methods=['GET','POST'])
@app.route('/home',methods=['GET','POST'])
def home():
    return render_template('index.html')


if __name__=="__main__":
    app.run(debug=True)