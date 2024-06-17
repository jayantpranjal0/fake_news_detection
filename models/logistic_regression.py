from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.linear_model import LogisticRegression 
import pandas as pd
from process import wordopt
import numpy as np
from joblib import dump
from joblib import load


def logistic_regression(x_train,y_train):
     #training for logistic regression
    vectorization = TfidfVectorizer() 
    x_train2 = vectorization.fit_transform(x_train) 
    LR = LogisticRegression() 
    LR.fit(x_train2, y_train) 
    dump(LR,'logistic_regression_model.joblib')
    dump(vectorization, 'tfidf_vectorization.joblib')

def manual_testing(news):
      #testing for logistic regression
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt) 
    new_x_test = new_def_test["text"] 
    vectorization = load('tfidf_vectorization.joblib')
    new_xv_test = vectorization.transform(new_x_test)
    LR = load('logistic_regression_model.joblib')
    pred_DT = LR.predict(new_xv_test)
    if pred_DT[0] ==0:
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
    LR = load('logistic_regression_model.joblib')
    new_xv_test = vectorization.transform(new_x_test)
    prob_DT = LR.predict_proba(new_xv_test)
    diff_DT=abs(prob_DT[0][1])
    return  f"{100*(diff_DT/(diff_DT +(1-diff_DT)*np.exp(-diff_DT)) )}"