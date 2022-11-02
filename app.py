from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import pickle
import numpy as np
from model import NLPModel

#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pickle
import pandas as pd
import tensorflow_hub as hub
#import tensorflow_text

#Preprocessing
#from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize



app = Flask(__name__)
api = Api(app)

#model = NLPModel()

clf_path = 'lib/models/use_lr.pkl'
with open(clf_path, 'rb') as f:
    mlb, p = pickle.load(f)

# argument parsing
parser = reqparse.RequestParser()
parser.add_argument('query')

#embed = hub.load("/Users/simon/Downloads/USE/")
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")


spec_chars = ["!",'"',"#","%","&","'","(",")",
              "*",",","-",".","/",":",";","<",
              "=",">","?","@","[","\\","]","^","_",
              "`","{","|","}","~","–", "$", "0", "1",
              "2", "3", "4", "5", "6", "7", "8", "9"]

class PredictSentiment(Resource):
    def get(self):
        # use parser and find the user's query
        args = parser.parse_args()
        user_query = args['query']
        print('LOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOL\n', user_query)

        ######################################################################
        #Preprocessing

        df = pd.DataFrame(data=[user_query], columns=['Body'])

        #df['Body'] = df['Body'].apply(lambda x: BeautifulSoup(x).get_text())

        df['Body'] = df['Body'].str.lower()

        for char in spec_chars:
            df['Body'] = df['Body'].str.replace(char, ' ')

        df['Body'] = df['Body'].str.split().str.join(" ")

        cachedStopWords = stopwords.words("english")

        df['Body'] = df['Body'].apply(lambda x: [str(word) for word in word_tokenize(x) if not word in cachedStopWords])
        df['Body'] = df['Body'].apply(lambda x: ' '.join(x))
        ######################################################################
        print('LOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOL\n', df['Body'])
        X_test_embed = df['Body'].to_list()
        X_test_embed = embed(X_test_embed)
        X_test_embed = np.array(X_test_embed)

        # Predict the classes using the pipeline
        mlb_predictions = p.predict(X_test_embed)
        #mlb_predictions = np.reshape(mlb_predictions, (20, mlb_predictions.shape[0]))
        # Turn those classes into labels using the binarizer.
        classes = mlb.inverse_transform(mlb_predictions)

        # create JSON object
        output = {'prediction': classes}

        return output


# Setup the Api resource routing here
# Route the URL to the resource
api.add_resource(PredictSentiment, '/')


if __name__ == '__main__':
    app.run(debug=True)
