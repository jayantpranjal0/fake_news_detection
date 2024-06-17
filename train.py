import pandas as pd 
from process import wordopt
from models.logistic_regression import logistic_regression
from models.decision_tree import decision_tree
from models.random_forest_classifier import random_forest
data = pd.read_csv(r"data/News.zip",index_col=0)
data = data.drop(["title", "subject","date"], axis = 1)
data = data.sample(frac=1) 
data.reset_index(inplace=True) 
data.drop(["index"], axis=1, inplace=True) 
data["text"] = data["text"].apply(wordopt)
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(data['text'],  
                                                    data['class'],  
                                                    test_size=0.25)
logistic_regression(x_train,y_train)
decision_tree(x_train,y_train)
random_forest(x_train,y_train)



