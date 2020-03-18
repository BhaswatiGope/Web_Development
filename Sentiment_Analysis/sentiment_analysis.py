# This is a model which reads movie review data and do sentiment analysis

import pandas as pd
import urllib.request
import os
import numpy as np
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
nltk.download('stopwords')

movie_data =pd.read_csv('movie_data.csv',encoding='utf-8')

# All standardization done on source data before training the model, tokenize the data and generate it.

tokenizer=RegexpTokenizer(r'\w+')
ps=PorterStemmer()
stopwords_english=set(stopwords.words('english'))


# Data has been stemmed and clean data got generated

def getReview(review):
    review=review.lower()
    review=review.replace("<br /><br />"," ")
    tokens=tokenizer.tokenize(review)
    tokens_generated=[token for token in tokens if token not in stopwords_english]
    tokens_stem=[ps.stem(token) for token in tokens_generated]
    review_clean_data=' '.join(tokens_stem)
    return review_clean_data

# Apply stemmed review on movie_data

movie_data['review'].apply(getReview)

# split data for train and test, considered 35000 for training and 15000 for testing

X_train = movie_data.loc[:35000, 'review'].values
y_train = movie_data.loc[:35000, 'sentiment'].values
X_test = movie_data.loc[35000:, 'review'].values
y_test = movie_data.loc[35000:, 'sentiment'].values

print (X_train.shape)
print (X_test.shape)


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer_tf = TfidfVectorizer(sublinear_tf=True, encoding='utf-8',decode_error='ignore')
vectorizer_tf.fit(X_train)
X_train=vectorizer_tf.transform(X_train)
X_test=vectorizer_tf.transform(X_test)

from sklearn.linear_model import LogisticRegression
model=LogisticRegression(solver='liblinear')
model.fit(X_train,y_train)
print("Training data score : "+str(model.score(X_train,y_train)))
print("Testing data score : "+str(model.score(X_test,y_test)))

model.predict(X_test[0])


model.predict_proba(X_test[0])
#array([[0.78833439, 0.21166561]])


from sklearn.externals import joblib
joblib.dump(stopwords_english, 'pkl_objects/stopwords.pkl')
joblib.dump(model, 'pkl_objects/model_sentiment.pkl')
joblib.dump(vectorizer_tf, 'pkl_objects/vectorizer_sentiment.pkl')