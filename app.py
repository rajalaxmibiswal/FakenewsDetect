import streamlit as st
import joblib

vectorizer = joblib.load("vectorizer.jb")
model = joblib.load("lr_model.jb")
st.title("Fake News Classifier")
st.write(" Enter a News Article below to check whether it is fake or real. ")

news_input = st.text_area("News Article Input:", "")

if st.button("check News"):
    if news_input.strip():
        transform_input = vectorizer.transform([news_input])
        prediction = model.predict(transform_input)
        if prediction[0] == 1:
            st.success("News is Real")
        else:
            st.error("News is Fake")
    else:
        st.warning("Please enter a news article !!!")
