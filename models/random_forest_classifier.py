from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from process import wordopt
import numpy as np
from joblib import dump
from joblib import load


def random_forest(x_train,y_train):
      #training for random forest classifier
    vectorization = TfidfVectorizer() 
    x_train2 = vectorization.fit_transform(x_train) 
    RF =RandomForestClassifier()
    RF.fit(x_train2, y_train) 
    dump(RF,'random_forest_classifier.joblib')
    dump(vectorization, 'tfidf_vectorization.joblib')

def manual_testing(news):
      #testing for logistic regression
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt) 
    new_x_test = new_def_test["text"] 
    vectorization = load('tfidf_vectorization.joblib')
    new_xv_test = vectorization.transform(new_x_test)
    RF = load('random_forest_classifier.joblib')
    pred_RF = RF.predict(new_xv_test)
    if pred_RF[0] ==0:
        return "Fake news"
    else:
        return "Not a Fake news"
def manual_prob(news):
    #calculating probabilty estimate of each class
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt) 
    new_x_test = new_def_test["text"]
    vectorization = load('tfidf_vectorization.joblib')
    RF = load('random_forest_classifier.joblib')
    new_xv_test = vectorization.transform(new_x_test)
    prob_RF = RF.predict_proba(new_xv_test)
    diff_RF=abs(prob_RF[0][1])
    return  f"{100*(diff_RF/(diff_RF +(1-diff_RF)*np.exp(-diff_RF)) )}"