import streamlit as st
import joblib as jl

vectorizer = jl.load('vectorizer.jb')
model = jl.load('lr_model.jb')
model = jl.load("lr_model.jb")

st.title("Fake News Detector")
st.write("Enter a News Article below to check if it is  Real or Fake.")

news_input=st.text_area("News Article:","")

if st.button("Check News"):
    if news_input.strip():
        transform_input = vectorizer.transform([news_input])
        prediction = model.predict(transform_input)

        if prediction[0] == 1:
            st.success("The news is **REAL**")
        else:
            st.error("The news is **FAKE**")
    else:
        st.warning("Please enter a news article to check.")
        

