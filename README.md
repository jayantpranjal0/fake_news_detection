# Fake News Detection System
## Introduction

This project is a basic implementation of fake news detection system. The model is trained on the dataset provided by Kaggle competition. The dataset contains 39000 news articles and their labels. The model is trained on the dataset and then used to predict the credibility of the news. The model is trained using Decision trees, Random forest and logistic regression. The model is then saved using joblib and used in the streamlit app to predict the credibility of the news. The project is configured to predict the credibility score of the news articles and also give a reasoning of the prediction. 

## Dataset
Dataset link: https://www.kaggle.com/datasets/jainpooja/fake-news-detection

## Installation
Clone the repository using 
```bash
git clone git@github.com:jayantpranjal0/fake_news_detection.git
```
Preferrably create a virtual environment using:
```bash
python -m venv env
```
Activate the virtual environment using:
```bash
source env/bin/activate

Create our API_KEY for accessing the GPT service
```

Install the dependencies using:
```bash
pip install -r requirements.txt
```
Train the required models using:
```bash
python train.py
```
Run the streamlit app using:
```bash
streamlit run app.py
```
