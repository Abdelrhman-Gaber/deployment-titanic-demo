import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px


classifier = joblib.load('hr_model.pkl')

st.title('HR Prediction App')
st.write('This app predicts whether an employee will be promoted or not based on their features.')
df = pd.read_csv('Hr promotion.csv')
def predict_promotion(department, region, education, gender, recruitment_channel, no_of_trainings, age, previous_year_rating, length_of_service, awards_won, avg_training_score):
    input_data = pd.DataFrame({
        'department': [department],
        'region': [region],
        'education': [education],
        'gender': [gender],
        'recruitment_channel': [recruitment_channel],
        'no_of_trainings': [no_of_trainings],
        'age': [age],
        'previous_year_rating': [previous_year_rating],
        'length_of_service': [length_of_service],
        'awards_won?': [awards_won],
        'avg_training_score': [avg_training_score]
    })
    label = ['Not Promoted', 'Promoted']
    prediction = classifier.predict(input_data)
    return label[prediction[0]]


def main():
    department = st.selectbox('Department', df['department'].unique())
    region = st.selectbox('Region', df['region'].unique())
    education = st.selectbox('Education', df['education'].unique())
    gender = st.selectbox('Gender', df['gender'].unique())
    recruitment_channel = st.selectbox('Recruitment Channel', df['recruitment_channel'].unique())
    no_of_trainings = st.number_input('Number of Trainings', min_value=0, max_value=10, value=0)
    age = st.number_input('Age', min_value=18, max_value=65, value=30)
    previous_year_rating = st.number_input('Previous Year Rating', min_value=1, max_value=5, value=3)
    length_of_service = st.number_input('Length of Service (years)', min_value=0, max_value=40, value=5)
    awards_won = st.selectbox('Awards Won?', [0, 1])
    avg_training_score = st.number_input('Average Training Score', min_value=0, max_value=100, value=50)

    if st.button('Predict Promotion'):
        result = predict_promotion(department, region, education, gender, recruitment_channel,
                                   no_of_trainings, age, previous_year_rating,
                                   length_of_service, awards_won, avg_training_score)
        st.success(f'The employee is predicted to be: {result}')

if __name__ == '__main__':
    main()           
