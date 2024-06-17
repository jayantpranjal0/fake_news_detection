from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.tree import DecisionTreeClassifier 
import pandas as pd
from process import wordopt
import numpy as np
from joblib import dump
from joblib import load


def decision_tree(x_train,y_train):
    #training for decision tree
    vectorization = TfidfVectorizer() 
    x_train2 = vectorization.fit_transform(x_train) 
    DT =DecisionTreeClassifier() 
    DT.fit(x_train2, y_train) 
    dump(DT,'decision_tree_model.joblib')
    dump(vectorization, 'tfidf_vectorization.joblib')

def manual_testing(news):
    #testing for decision tree
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt) 
    new_x_test = new_def_test["text"] 
    vectorization = load('tfidf_vectorization.joblib')
    new_xv_test = vectorization.transform(new_x_test)
    DT = load('decision_tree_model.joblib')
    pred_LR = DT.predict(new_xv_test)
    if pred_LR[0] ==0:
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
    DT = load('decision_tree_model.joblib')
    new_xv_test = vectorization.transform(new_x_test)
    prob_LR = DT.predict_proba(new_xv_test)
    diff_LR=abs(prob_LR[0][1])
    return  f"{100*(diff_LR/(diff_LR +(1-diff_LR)*np.exp(-diff_LR)) )}"