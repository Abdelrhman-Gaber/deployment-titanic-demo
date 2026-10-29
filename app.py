
import streamlit as st
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np


classifier = joblib.load('titanic_model.pkl')

def predict_survival(pclass, sex, age, sibsp, parch, fare, embarked):
    input_data = pd.DataFrame({
        'pclass': [pclass],
        'sex': [sex],
        'age': [age],
        'sibsp': [sibsp],
        'parch': [parch],
        'fare': [fare],
        'embarked': [embarked]
    })
    prediction = classifier.predict(input_data)
    label = 'Survived' if prediction[0] == 1 else 'Did not survive'
    return label

def main():
    st.title('Titanic prediction')
    html_temp="""
                <div style="background-color:red">
                <h2 style="color:white;text-align:center;">this our streamlit Deployment App</h2>
                </div>
              """
    st.markdown(html_temp,unsafe_allow_html=True)

    pclass = st.selectbox("Passenger Class (1, 2, 3)", [1, 2, 3])
    sex = st.selectbox("Sex", ['male', 'female'])
    age = st.number_input("Age", min_value=0, max_value=100, value=30)
    sibsp = st.number_input("Number of Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
    parch = st.number_input("Number of Parents/Children Aboard", min_value=0, max_value=10, value=0)
    fare = st.number_input("Fare", min_value=0.0, max_value=500.0, value=32.2)
    embarked = st.selectbox("Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)", ['C', 'Q', 'S'])

    if st.button("Predict"):
        result = predict_survival(pclass, sex, age, sibsp, parch, fare, embarked)
        st.success(f"The passenger would have: {result}")    


if __name__ == '__main__':  
    main()
