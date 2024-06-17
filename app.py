from models.logistic_regression import manual_testing,manual_prob
from views import ask_gpt
import streamlit as st
from models import decision_tree as DT,logistic_regression as LR,random_forest_classifier as RF

st.title("Fake News Detector")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"]=="user":
            st.markdown(message["content"])
        else:
            st.write(f"Credibilty Score is :**{message['content']['avgscore']}**")
            with st.spinner("Analysis loading"):
                st.markdown(message["content"]["text"].replace("###",'')) 

if prompt := st.chat_input("Enter the News?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        score_LR=LR.manual_prob(prompt)
        score_DT=DT.manual_prob(prompt)
        score_RF=RF.manual_prob(prompt)
        avg_score=(float(score_LR)+float(score_DT)+float(score_RF))/3.0
        st.write(f"Credibilty Score is :**{avg_score}**")
        with st.spinner("Analysis Loading"):
            text=ask_gpt(prompt,DT.manual_testing(prompt))
            st.markdown(text.replace("###",'')) 
    st.session_state.messages.append({"role": "assistant", "content": {"avgscore":avg_score,"text":text}})
