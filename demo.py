# -*- coding: utf-8 -*-

__author__ = 'xead'
from sentiment_classifier import SentimentClassifier
from sentiment_classifier import LemmaTokenizer
from codecs import open
import time
from flask import Flask, render_template, request, url_for
import pandas as pd
import os

app = Flask(__name__)

print "Preparing classifier"
start_time = time.time()
classifier = SentimentClassifier()
print "Classifier is ready"
print time.time() - start_time, "seconds"

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
test_data = pd.read_csv(os.path.join(APP_ROOT, 'products_sentiment_test.tsv'), sep='\t') 
test_data['links'] = test_data.apply(lambda row: '<a href="custom?id=' + str(row.name) + '">' + row['text'] + '</a>', axis=1)
pd.set_option('display.max_colwidth', -1)

@app.route("/", methods=["POST", "GET"])
@app.route("/custom", methods=["POST", "GET"])
def index_page(text="", preprocessed_text="", prediction_message="", feature_weights=""):  
    
    if request.method == "POST":
        text = request.form.get('text')
    else:
        id = request.args.get("id")
        print id
        if id is not None:
            text = test_data.loc[int(id)]['text']

    if text is not None and len(text) > 0:
        preprocessed_text = classifier.preprocess(text)
        vectorized = classifier.vectorize(text)
        prediction_message = classifier.get_prediction_message(vectorized)
        feature_weights = classifier.get_feature_weights(vectorized)
    
    return render_template('custom.html', text=text, prediction_message=prediction_message, preprocessed_text=preprocessed_text, feature_weights=feature_weights)

@app.route("/samples", methods=["GET"])
def samples(text="", prediction_message=""):    
    return render_template('sample.html', tables=[test_data.to_html(columns=['links'], header=False, index=False, escape=False)], text=text, prediction_message=prediction_message)

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)    
